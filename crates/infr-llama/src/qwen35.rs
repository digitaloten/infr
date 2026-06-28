//! Qwen3.5 / Qwen3.6 (`qwen35`, aka Qwen3-Next): hybrid gated-DeltaNet linear-attention + gated
//! full-attention. See `docs/QWEN35.md`. This module is a **CPU reference** (correctness first);
//! the GPU path comes after the math is locked against llama.cpp.
#![allow(dead_code)] // forward pass is built up incrementally on this loader

use crate::load_f32;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// Parsed `qwen35` hyper-parameters (subset needed for the 0.8B dense model).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub eps: f32,
    // attention layers
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize, // key_length == value_length (256)
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rope_sections: [u32; 4],
    pub full_attn_interval: usize,
    // linear (gated DeltaNet) layers
    pub d_conv: usize,  // ssm conv kernel (4)
    pub d_state: usize, // head_k_dim (128)
    pub d_inner: usize, // value_dim (2048)
    pub n_group: usize, // num_k_heads (16)
    pub dt_rank: usize, // num_v_heads (16)
}

impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != "qwen35" {
            bail!("not a qwen35 model (arch={arch:?})");
        }
        let u = |k: &str| g.metadata().u64(&format!("qwen35.{k}"));
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("missing qwen35.{k}"));
        let f = |k: &str| -> Option<f32> {
            g.metadata()
                .get(&format!("qwen35.{k}"))
                .and_then(|v| match v {
                    infr_core::MetaValue::F64(x) => Some(*x as f32),
                    infr_core::MetaValue::U64(x) => Some(*x as f32),
                    infr_core::MetaValue::I64(x) => Some(*x as f32),
                    _ => None,
                })
        };
        // rope.dimension_sections is an array [11,11,10,0]
        let sections: [u32; 4] = {
            let mut s = [0u32; 4];
            if let Some(arr) = g
                .metadata()
                .get("qwen35.rope.dimension_sections")
                .and_then(|v| v.as_arr())
            {
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u64().unwrap_or(0) as u32;
                }
            }
            s
        };
        Ok(Cfg {
            n_layer: req("block_count")? as usize,
            n_embd: req("embedding_length")? as usize,
            vocab: 0, // filled from token_embd shape
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            n_head: req("attention.head_count")? as usize,
            n_kv: req("attention.head_count_kv")? as usize,
            head_dim: req("attention.key_length")? as usize,
            rope_dim: u("rope.dimension_count").unwrap_or(64) as usize,
            rope_theta: f("rope.freq_base").unwrap_or(1e7),
            rope_sections: sections,
            full_attn_interval: u("full_attention_interval").unwrap_or(4) as usize,
            d_conv: req("ssm.conv_kernel")? as usize,
            d_state: req("ssm.state_size")? as usize,
            d_inner: req("ssm.inner_size")? as usize,
            n_group: req("ssm.group_count")? as usize,
            dt_rank: req("ssm.time_step_rank")? as usize,
        })
    }

    /// Attention (vs linear/SSM) layer test: every `full_attn_interval`-th layer is full attention.
    pub fn is_attn_layer(&self, i: usize) -> bool {
        (i + 1) % self.full_attn_interval == 0
    }
    pub fn num_k_heads(&self) -> usize {
        self.n_group
    }
    pub fn num_v_heads(&self) -> usize {
        self.dt_rank
    }
    pub fn head_k_dim(&self) -> usize {
        self.d_state
    }
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.dt_rank
    }
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
}

/// A linear (gated DeltaNet) layer's weights, all dequantized to f32.
struct LinearLayer {
    attn_norm: Vec<f32>, // [n_embd]
    qkv: Vec<f32>,       // [conv_channels, n_embd]  (out,in)
    gate: Vec<f32>,      // [d_inner, n_embd]  (z)
    conv1d: Vec<f32>,    // [conv_channels, d_conv]  (per-channel kernel)
    alpha: Vec<f32>,     // [dt_rank, n_embd]
    beta: Vec<f32>,      // [dt_rank, n_embd]
    a: Vec<f32>,         // [dt_rank]  (= -exp(A_log))
    dt_bias: Vec<f32>,   // [dt_rank]
    ssm_norm: Vec<f32>,  // [head_v_dim]
    out: Vec<f32>,       // [n_embd, d_inner]
    post_norm: Vec<f32>, // [n_embd]
    ffn_gate: Vec<f32>,  // [n_ff, n_embd]
    ffn_up: Vec<f32>,    // [n_ff, n_embd]
    ffn_down: Vec<f32>,  // [n_embd, n_ff]
    n_ff: usize,
}

/// A full-attention layer's weights.
struct AttnLayer {
    attn_norm: Vec<f32>, // [n_embd]
    q: Vec<f32>,         // [n_head*head_dim + d_inner(gate), n_embd]
    k: Vec<f32>,         // [n_kv*head_dim, n_embd]
    v: Vec<f32>,         // [n_kv*head_dim, n_embd]
    q_norm: Vec<f32>,    // [head_dim]
    k_norm: Vec<f32>,    // [head_dim]
    out: Vec<f32>,       // [n_embd, n_head*head_dim]
    post_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    n_ff: usize,
}

enum Layer {
    Linear(LinearLayer),
    Attn(AttnLayer),
}

/// Full model weights (CPU, f32).
pub struct Model {
    pub cfg: Cfg,
    token_embd: Vec<f32>, // [vocab, n_embd]
    output_norm: Vec<f32>,
    lm_head: Vec<f32>, // [vocab, n_embd]
    layers: Vec<Layer>,
}

impl Model {
    pub fn load(g: &Gguf) -> Result<Self> {
        let mut cfg = Cfg::from_gguf(g)?;
        let (token_embd, te_shape) = load_f32(g, "token_embd.weight")?;
        cfg.vocab = te_shape[1];
        let output_norm = load_f32(g, "output_norm.weight")?.0;
        let lm_head = if g.tensors().iter().any(|t| t.name == "output.weight") {
            load_f32(g, "output.weight")?.0
        } else {
            token_embd.clone() // tied
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let get = |s: &str| -> Result<Vec<f32>> {
                load_f32(g, &p(s)).map(|x| x.0).with_context(|| p(s))
            };
            let ffn_up_shape = g
                .tensors()
                .iter()
                .find(|t| t.name == p("ffn_up.weight"))
                .map(|t| t.shape.clone())
                .context("ffn_up")?;
            let n_ff = ffn_up_shape[1];
            if cfg.is_attn_layer(i) {
                layers.push(Layer::Attn(AttnLayer {
                    attn_norm: get("attn_norm.weight")?,
                    q: get("attn_q.weight")?,
                    k: get("attn_k.weight")?,
                    v: get("attn_v.weight")?,
                    q_norm: get("attn_q_norm.weight")?,
                    k_norm: get("attn_k_norm.weight")?,
                    out: get("attn_output.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: get("ffn_gate.weight")?,
                    ffn_up: get("ffn_up.weight")?,
                    ffn_down: get("ffn_down.weight")?,
                    n_ff,
                }));
            } else {
                layers.push(Layer::Linear(LinearLayer {
                    attn_norm: get("attn_norm.weight")?,
                    qkv: get("attn_qkv.weight")?,
                    gate: get("attn_gate.weight")?,
                    conv1d: get("ssm_conv1d.weight")?,
                    alpha: get("ssm_alpha.weight")?,
                    beta: get("ssm_beta.weight")?,
                    a: get("ssm_a")?,
                    dt_bias: get("ssm_dt.bias")?,
                    ssm_norm: get("ssm_norm.weight")?,
                    out: get("ssm_out.weight")?,
                    post_norm: get("post_attention_norm.weight")?,
                    ffn_gate: get("ffn_gate.weight")?,
                    ffn_up: get("ffn_up.weight")?,
                    ffn_down: get("ffn_down.weight")?,
                    n_ff,
                }));
            }
        }
        Ok(Model {
            cfg,
            token_embd,
            output_norm,
            lm_head,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        // the 0.8B pulled into our store
        dirs_cache().join("infr/models/blobs/sha256-bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517")
    }
    fn dirs_cache() -> std::path::PathBuf {
        std::env::var("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache")
            })
    }

    #[test]
    #[ignore = "needs the Qwen3.5-0.8B gguf in the local store"]
    fn loads_and_dims() {
        let g = Gguf::open(&model_path()).unwrap();
        let m = Model::load(&g).unwrap();
        let c = &m.cfg;
        println!("cfg: {c:?}");
        println!(
            "k_heads={} head_k={} v_heads={} head_v={} conv_ch={}",
            c.num_k_heads(),
            c.head_k_dim(),
            c.num_v_heads(),
            c.head_v_dim(),
            c.conv_channels()
        );
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.conv_channels(), 6144);
        assert_eq!(c.head_v_dim(), 128);
        let n_attn = (0..c.n_layer).filter(|&i| c.is_attn_layer(i)).count();
        assert_eq!(n_attn, 6, "expected 6 full-attention layers");
        assert_eq!(m.layers.len(), 24);
    }
}
