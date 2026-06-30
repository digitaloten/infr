//! Minimal autoregressive **Llama** inference for GGUF models, for fast GPU bring-up.
//!
//! Strategy (bring-up): the heavy linear projections run on the GPU (`infr-vulkan` eager
//! `linear`, weights uploaded once); the cheap ops (embedding gather, RMSNorm, RoPE, GQA
//! attention, SwiGLU, residual, sampling) run on the host. No KV cache yet — each step does a
//! full-prefix forward (fine for a tiny model). Validated on SmolLM2-135M.
//!
//! TODO(next): move host ops to GPU; add a KV cache; fold into the `Model`/`Backend` seams.
#![allow(clippy::needless_range_loop)]

mod config;
pub mod cpu_backend;
mod transformer;
pub(crate) use transformer::PerLayerEmbd;
pub use transformer::{ChatSession, Llama};
mod kv;
pub(crate) use kv::{DecodeScratch, DenseDecodeScratch, PrefillScratch, QBufs};
pub use kv::{KvCache, MoeConfig, MoeKv};
mod sampling;
pub use config::Config;
pub(crate) use sampling::*;
mod weights;
pub(crate) use weights::*;
pub use weights::{weight_footprint, WeightFootprint};
mod mixers;
pub mod model;
mod quant;
pub mod qwen35;
mod tokenizer;
pub(crate) use quant::*;
pub(crate) use tokenizer::*;

use anyhow::{anyhow, Result};
use infr_chat::render_chat_user;
use infr_core::WeightSource;
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use std::path::Path;
use tokenizers::Tokenizer;

/// Qwen2/Qwen3 pre-tokenizer regex (same string the HF `tokenizer.json` uses) — applied via a
/// Split before ByteLevel. Differs from the default GPT-2 ByteLevel regex (punctuation/number runs),
/// which is what made a naive ByteLevel produce different token ids.
pub(crate) const QWEN2_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Build the gemma4 E2B per-layer-embedding global tensors from the GGUF (host f32 — no GPU). The
/// big `per_layer_token_embd` stays quantized in the mmap and is gathered per token at forward time.
/// `None` for models without per-layer embeddings. Shared by the GPU and CPU loaders.
fn build_per_layer_embd(g: &Gguf, cfg: &Config) -> Result<Option<PerLayerEmbd>> {
    if cfg.n_embd_per_layer == 0 {
        return Ok(None);
    }
    let (model_proj, _) = load_tensor_dequant(g, "per_layer_model_proj.weight")?;
    let (proj_norm, _) = load_tensor_dequant(g, "per_layer_proj_norm.weight")?;
    let te = g
        .tensors()
        .iter()
        .find(|t| t.name == "per_layer_token_embd.weight")
        .ok_or_else(|| anyhow!("per_layer_token_embd.weight not found"))?;
    // Bytes per token row = total bytes / vocab (te shape is GGUF [npl*n_layer, vocab]).
    let te_vocab = *te.shape.last().unwrap();
    Ok(Some(PerLayerEmbd {
        npl: cfg.n_embd_per_layer,
        n_layer: cfg.n_layer,
        n_embd: cfg.n_embd,
        model_proj,
        proj_norm,
        tok_embd_dtype: te.dtype,
        tok_embd_row_bytes: te.nbytes / te_vocab,
    }))
}

/// UTF-8-safe incremental detokenizer for streaming: appends `id` to `acc`, decodes the whole
/// sequence so far, and emits the newly-completed suffix past `printed` — holding back a trailing
/// `�` (a multi-byte char split across tokens) until it completes. Mirrors the GPU path's streamer.
fn stream_token(
    tokenizer: &Tokenizer,
    acc: &mut Vec<u32>,
    printed: &mut usize,
    id: u32,
    on_piece: &mut impl FnMut(&str),
) {
    acc.push(id);
    if let Ok(full) = tokenizer.decode(acc, true) {
        if !full.ends_with('\u{FFFD}') && full.len() > *printed && full.is_char_boundary(*printed) {
            on_piece(&full[*printed..]);
            *printed = full.len();
        }
    }
}

// Chat-template rendering (`render_chat_jinja`, `render_chat_user`) lives in the shared `infr-chat`
// crate — imported at the top of this module. There is NO fabricated-ChatML fallback: infr supports
// only models that ship a `tokenizer.chat_template`, so a missing/broken template is a hard error.

/// The error surfaced when a GGUF has no usable chat template (none embedded, or it failed to render).
fn no_template_err() -> anyhow::Error {
    anyhow!(
        "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
         render — set INFR_DEBUG_CHAT=1 for details). infr requires an instruct model with an \
         embedded chat template."
    )
}

/// Whether a Vulkan device is available — a cheap probe (creates and drops a backend). Lets callers
/// (and tests) decide between the GPU and CPU paths, or skip GPU-only work when there's no device.
pub fn gpu_available() -> bool {
    VulkanBackend::new().is_ok()
}

/// Locate the Qwen3-0.6B Q4_K_M GGUF in the HF Hub cache (or `INFR_TEST_MODEL`) for the model-backed
/// unit tests; `None` → the test self-skips. We use the shared HF cache everywhere now (no bespoke
/// local model dir).
#[cfg(test)]
fn test_qwen3_06b() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join("Qwen3-0.6B-Q4_K_M.gguf");
        f.exists().then_some(f)
    })
}

/// Append chat-end markers in the vocab (`<|im_end|>` / `<|endoftext|>` / `<|eot_id|>`) to
/// `cfg.eos_ids` so generation stops on any of them, not just the GGUF `eos`.
fn add_chat_eos(cfg: &mut Config, tokenizer: &Tokenizer) {
    for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
        if let Some(id) = tokenizer.token_to_id(name) {
            if !cfg.eos_ids.contains(&id) {
                cfg.eos_ids.push(id);
            }
        }
    }
}

/// A **GPU-free** model for the CPU reference backend. Holds only what the agnostic CPU compute
/// graph needs — the parsed [`Config`], the host f32 token embeddings (for the gather + tied lm
/// head), the tokenizer, and the gemma4 E2B per-layer-embd tensors. No `VulkanBackend`, no VRAM,
/// no weight upload: the projection weights are streamed straight from the kept-open GGUF mmap at
/// forward time. Dense Qwen3/Llama, Gemma 3, Gemma 4 (dense + E2B), and qwen3moe; for qwen35 use
/// [`crate::qwen35::generate_cpu`].
pub struct CpuModel {
    gguf: Gguf,
    cfg: Config,
    token_embd: Vec<f32>,
    per_layer_embd: Option<PerLayerEmbd>,
    tokenizer: Tokenizer,
}

impl CpuModel {
    /// Load a model for CPU inference without touching the GPU. `tokenizer_path` overrides the
    /// GGUF's embedded vocab when given.
    pub fn load(gguf_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let g = Gguf::open(gguf_path).map_err(|e| anyhow!("open gguf: {e}"))?;
        let mut cfg = Config::from_gguf(&g)?;
        let tokenizer = match tokenizer_path {
            Some(p) => Tokenizer::from_file(p).map_err(|e| anyhow!("load tokenizer: {e}"))?,
            None => build_tokenizer(&g)?,
        };
        add_chat_eos(&mut cfg, &tokenizer);
        let (token_embd, _) = load_tensor_dequant(&g, "token_embd.weight")?;
        let per_layer_embd = build_per_layer_embd(&g, &cfg)?;
        Ok(Self {
            gguf: g,
            cfg,
            token_embd,
            per_layer_embd,
            tokenizer,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Render a user turn with the model's OWN embedded chat template (so an instruct model — Gemma,
    /// Qwen, … — answers coherently). Errors if the GGUF has no `tokenizer.chat_template` or it fails
    /// to render — infr only supports models that ship one (no fabricated-ChatML fallback).
    pub fn render_chat(&self, user: &str) -> Result<String> {
        render_chat_user(&self.gguf, &self.tokenizer, self.cfg.eos, user)
            .ok_or_else(no_template_err)
    }

    /// Greedy generation on the CPU reference backend (no GPU). Returns the decoded text plus
    /// timing/counts ([`crate::cpu_backend::CpuStats`]) for the caller's stats line.
    /// The generated text is delivered through `on_piece` as it streams; only timing/counts are
    /// returned.
    pub fn generate_cpu(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_piece: impl FnMut(&str),
    ) -> Result<crate::cpu_backend::CpuStats> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let prompt_tokens: Vec<u32> = enc.get_ids().to_vec();
        // Stream each generated token: incrementally detokenize and emit the new suffix.
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (_generated, stats) = crate::cpu_backend::generate_dense_cpu(
            &self.gguf,
            &self.cfg,
            &self.token_embd,
            self.per_layer_embd.as_ref(),
            &prompt_tokens,
            max_new,
            |id| stream_token(&self.tokenizer, &mut acc, &mut printed, id, &mut on_piece),
        )?;
        Ok(stats)
    }
}

pub(crate) fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata().u64(key)
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    // ── IQ4_NL ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]], 32 elements, 18 bytes
    // y[j] = d * KVALUES_IQ4NL[qs[j] & 0xF]; y[j+16] = d * KVALUES_IQ4NL[qs[j] >> 4]
    // Reference: llama.cpp dequantize_row_iq4_nl (ggml-quants.c l.2653)
    #[test]
    fn iq4nl_single_block() {
        // d=1.0, qs[0]=0x80 (lo=0, hi=8)
        // KVALUES_IQ4NL[0] = -127, KVALUES_IQ4NL[8] = 1
        // y[0] = 1.0 * (-127) = -127.0
        // y[16] = 1.0 * 1 = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x80; // lo=0→-127, hi=8→1
        let y = dequant_codebook(infr_core::DType::Iq4Nl, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - (-127.0)).abs() < 1e-3,
            "iq4nl y[0] expected -127.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 1.0).abs() < 1e-3,
            "iq4nl y[16] expected 1.0, got {}",
            y[16]
        );
    }

    // ── IQ4_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 scales_h][uint8 scales_l[4]][uint8 qs[128]], 256 elements, 136 bytes
    // y = d*(ls-32) * KVALUES_IQ4NL[q4], ls is 6-bit per 32-elem sub-block
    // Reference: llama.cpp dequantize_row_iq4_xs (ggml-quants.c l.2671)
    #[test]
    fn iq4xs_single_block() {
        // d=1.0, scales: all sub-blocks have ls=32 → dl=d*(32-32)=0 → y=0
        // Verify: all 256 outputs are 0.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&d_bytes);
        // scales_h=0, scales_l=[0x00,0x00,0x00,0x00]: all lo=0, all hi=0 → ls=0 → dl=-32
        // Wait: ls=lo|(hi<<4). With scales_h=0 and scales_l=0, ls=0. dl=1.0*(0-32)=-32.
        // qs all 0: qs[j]&0xF=0 → KVALUES_IQ4NL[0]=-127; qs[j]>>4=0 → -127
        // y = -32 * (-127) = 4064.0 (all elements)
        let y = dequant_codebook(infr_core::DType::Iq4Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = -32.0_f32 * KVALUES_IQ4NL[0] as f32; // 4064.0
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 0.5,
                "iq4xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ1_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[32]][u16 qh[8]], 50 bytes, QK_K=256
    // All-zero block: d=1.0, qh=0 → dl=1.0*(2*0+1)=1.0, delta=+0.125, grid_idx=0
    //   IQ1S_GRID[0] = 0xffffffffffffffff → gv=-1 for all j
    //   y[j] = 1.0 * (-1.0 + 0.125) = -0.875 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq1_s (ggml-quants.c l.2578)
    #[test]
    fn iq1s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&d_bytes);
        // qs=0, qh=0 → grid_idx=0, dl=1.0, delta=+0.125
        let y = dequant_codebook(infr_core::DType::Iq1S, &block);
        assert_eq!(y.len(), 256);
        // IQ1S_GRID[0] = 0xffffffffffffffff → all bytes 0xFF = -1i8
        let expected = 1.0_f32 * (-1.0_f32 + IQ1S_DELTA);
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq1s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── MXFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 e][u8 qs[16]], 17 bytes, QK_MXFP4=32
    // e=128 → d=e8m0_to_fp32_half(128)=2^(128-128)=1.0; qs[0]=0x21 → lo=1, hi=2
    //   y[0] = KVALUES_MXFP4[1]*1.0 = 1.0; y[16] = KVALUES_MXFP4[2]*1.0 = 2.0
    // Ref: llama.cpp dequantize_row_mxfp4 (ggml-quants.c l.511)
    #[test]
    fn mxfp4_single_block() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e=128 → d=1.0
        block[1] = 0x21; // lo nibble=1→1, hi nibble=2→2
        let y = dequant_codebook(infr_core::DType::Mxfp4, &block);
        assert_eq!(y.len(), 32);
        assert!(
            (y[0] - 1.0).abs() < 1e-5,
            "mxfp4 y[0] expected 1.0, got {}",
            y[0]
        );
        assert!(
            (y[16] - 2.0).abs() < 1e-5,
            "mxfp4 y[16] expected 2.0, got {}",
            y[16]
        );
        // rest of qs=0 → x0=x1=0 → y=0.0
        for i in 1..16 {
            assert!(y[i].abs() < 1e-5, "mxfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── NVFP4 ───────────────────────────────────────────────────────────────────
    // Block: [u8 d[4]][u8 qs[32]], 36 bytes, QK_NVFP4=64
    // All-zero scales: d=ue4m3_to_fp32(0)=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_nvfp4 (ggml-quants.c l.531)
    #[test]
    fn nvfp4_single_block() {
        let block = vec![0u8; 36];
        let y = dequant_codebook(infr_core::DType::Nvfp4, &block);
        assert_eq!(y.len(), 64);
        for i in 0..64 {
            assert!(y[i].abs() < 1e-5, "nvfp4 y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── IQ1_M ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[32]][u8 qh[16]][u8 scales[8]], 56 bytes, QK_K=256
    // All-zero: scales=0 → d_bits=0 → d=0.0 → all y=0.0
    // Ref: llama.cpp dequantize_row_iq1_m (ggml-quants.c l.2603)
    #[test]
    fn iq1m_single_block() {
        let block = vec![0u8; 56];
        let y = dequant_codebook(infr_core::DType::Iq1M, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(y[i].abs() < 1e-4, "iq1m y[{i}] expected 0.0, got {}", y[i]);
        }
    }

    // ── TQ1_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[48]][u8 qh[4]][half d], 54 bytes, QK_K=256
    // All-zero qs/qh: q=0 → xi=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq1_0 (ggml-quants.c l.2356)
    #[test]
    fn tq1_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 54];
        block[52..54].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq1_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq1_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── TQ2_0 ───────────────────────────────────────────────────────────────────
    // Block: [u8 qs[64]][half d], 66 bytes, QK_K=256
    // All-zero qs: q=(0>>l*2)&3=0 → y=(0-1)*d = -d for all 256 elements
    // Ref: llama.cpp dequantize_row_tq2_0 (ggml-quants.c l.2395)
    #[test]
    fn tq2_0_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[64..66].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Tq2_0, &block);
        assert_eq!(y.len(), 256);
        for i in 0..256 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "tq2_0 y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]], 66 bytes, QK_K=256
    // Sub-block 0: aux0=0 → 4 grid indices all 0; aux1=0 → scale_mag=0, sign_idx=0
    //   IQ2XXS_GRID[0] = 0x0808080808080808 → 8 bytes all 0x08
    //   KSIGNS_IQ2XS[0] = 0 → no negations
    //   db = 1.0*(0.5+0)*0.25 = 0.125
    //   y = 0.125 * 8 = 1.0 for each of 8 elements × 4 groups = 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xxs (ggml-quants.c l.2416)
    #[test]
    fn iq2xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 66];
        block[0..2].copy_from_slice(&d_bytes);
        // all qs = 0: aux0=0 (grid_idx=0), aux1=0 (scale_mag=0, sign_idx=0)
        let y = dequant_codebook(infr_core::DType::Iq2Xxs, &block);
        assert_eq!(y.len(), 256);
        // first sub-block, first element
        let expected = 0.125 * 8.0_f32;
        for i in 0..32 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
        // remaining sub-blocks: same pattern (all zeros)
        for i in 32..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_XS ──────────────────────────────────────────────────────────────────
    // Block: [half d][uint16 qs[32]][uint8 scales[8]], 74 bytes, QK_K=256
    // All zeros: scales[0]=0 → db0=db1=0.125; qs16=0 → grid_idx=0, sign_idx=0
    //   IQ2XS_GRID[0] = 0x0808080808080808 → gv=8; KSIGNS[0]=0 → +1
    //   y = 0.125 * 8 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq2_xs (ggml-quants.c l.2444)
    #[test]
    fn iq2xs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2Xs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2xs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ2_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 scales[8]], 82 bytes, QK_K=256
    // All zeros: scales=0 → db0=db1=0.125; qs_all[0]=0, qh[0]=0 → grid_idx=0
    //   IQ2S_GRID[0] = 0x0808080808080808 → gv=8; signs[32]=0 → +1
    //   y = 0.125 * 8 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq2_s (ggml-quants.c l.2471)
    #[test]
    fn iq2s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq2S, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.125 * 8.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq2s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_XXS ─────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[96]], 98 bytes, QK_K=256
    // qs[0..64]=0 → grid_idx=0 for all; qs[64..96]=0 → aux32=0 → scale_mag=0, sign_idx=0
    //   IQ3XXS_GRID[0] = 0x04040404 → gv for j=0..3: 4; KSIGNS[0]=0 → +1
    //   db = 1.0*(0.5+0)*0.5 = 0.25
    //   y = 0.25 * 4 = 1.0 for first 32 elements
    // Ref: llama.cpp dequantize_row_iq3_xxs (ggml-quants.c l.2503)
    #[test]
    fn iq3xxs_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3Xxs, &block);
        assert_eq!(y.len(), 256);
        let expected = 0.25 * 4.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3xxs y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── IQ3_S ───────────────────────────────────────────────────────────────────
    // Block: [half d][u8 qs[64]][u8 qh[8]][u8 signs[32]][u8 scales[4]], 110 bytes
    // All zeros: scales=0 → db1=db2=1.0*(1+2*0)=1.0; qs=0, qh=0 → grid_idx=0
    //   IQ3S_GRID[0] = 0x01010101 → gv for j=0..3: 1; signs[0]=0 → +1
    //   y = 1.0 * 1 = 1.0 for all 256 elements
    // Ref: llama.cpp dequantize_row_iq3_s (ggml-quants.c l.2535)
    #[test]
    fn iq3s_single_block() {
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&d_bytes);
        let y = dequant_codebook(infr_core::DType::Iq3S, &block);
        assert_eq!(y.len(), 256);
        let expected = 1.0_f32;
        for i in 0..256 {
            assert!(
                (y[i] - expected).abs() < 1e-4,
                "iq3s y[{i}] expected {expected}, got {}",
                y[i]
            );
        }
    }

    // ── Q2_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 scales[16]][uint8 qs[64]][half d][half dmin]
    // y = d*(sc&0xF)*q2 - dmin*(sc>>4), q2 ∈ 0..3
    // Reference: llama.cpp dequantize_row_q2_K (ggml-quants.c l.903)
    #[test]
    fn q2k_single_block() {
        // d=1.0, dmin=2.0
        // scales[0]=0x23 → lo=3, hi=2 → first sub-block: dl=3.0, ml=4.0
        // scales[1]=0x23 → second 16-elem sub-block (qs[16..32]): same dl/ml
        // qs[0..16]=0x55 → q2 (shift=0) = 0x55 & 3 = 1
        // Expected y[0] = 3.0*1 - 4.0 = -1.0
        let mut block = vec![0u8; 84];
        // scales[0..16]
        block[0] = 0x23; // lo=3, hi=2
        block[1] = 0x23; // same for second sub-block
                         // qs[16..80]
        for b in &mut block[16..80] {
            *b = 0x55; // any bits; q2 at shift=0 for first 16 = 1
        }
        // d[80..82] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[80..82].copy_from_slice(&d_bytes);
        // dmin[82..84] = 2.0
        let dmin_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        block[82..84].copy_from_slice(&dmin_bytes);

        let y = dequant_block(infr_core::DType::Q2K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // First sub-block, first element: q2=1, y=3.0*1-4.0=-1.0
        assert!(
            (y[0] - (-1.0)).abs() < 1e-4,
            "q2k y[0] expected -1.0, got {}",
            y[0]
        );
        // All elements in first sub-block same q2=1 → same y
        for i in 0..16 {
            assert!(
                (y[i] - (-1.0)).abs() < 1e-4,
                "q2k y[{i}] expected -1.0, got {}",
                y[i]
            );
        }
        // Second sub-block (16..32): same scales, qs=0x55, q2=(0x55>>2)&3=(0x15)&3=1
        // Wait: shift=0 for j=0 applies to BOTH first and second 16-elem groups of the
        // same j-iteration. Let me re-check the llama logic.
        // In the llama code, for j=0 (shift=0):
        //   sc=scales[0], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l] >> 0) & 3 = qs[l] & 3 = 0x55 & 3 = 1
        //   sc=scales[1], dl=d*(sc&0xF)=3, ml=dmin*(sc>>4)=4
        //   for l in 0..16: q2 = (q[l+16] >> 0) & 3 = qs[l+16] & 3 = 1
        // So elements 16..32 also have dl=3, ml=4, q2=1 → y=-1.0
        assert!(
            (y[16] - (-1.0)).abs() < 1e-4,
            "q2k y[16] expected -1.0, got {}",
            y[16]
        );
    }

    // ── Q3_K ────────────────────────────────────────────────────────────────────
    // Block: [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][half d]
    // y = d*(sc6-32)*(q3u - 4), q3u = (low2 | high_bit<<2) ∈ 0..7
    // Reference: llama.cpp dequantize_row_q3_K (ggml-quants.c l.1247)
    #[test]
    fn q3k_single_block() {
        // d=1.0
        // Choose scales to decode as sc6=36 for all sub-blocks → sc6-32=4 → dl=4.0
        // Encode sc6=36 for first sub-block in scales_raw:
        //   After decode, aux bytes give sc6 values. Simpler: set all scales[0..12]=0
        //   so that after bit manipulation aux has all-zero lower nibbles → sc6=0 for all.
        //   Then dl=0 → y=0 everywhere. That's a trivial test.
        //
        // Better: set scales bytes to give sc6=32 for first two sub-blocks (dl=0, y=0)
        // and verify that y[0..32]=0. Then set hmask and qs to anything.
        //
        // Even simpler: set scales_raw all-zero. After bit manipulation:
        //   aux[0]=0, aux[1]=0, aux[2]=0, aux[3]=0
        //   sc6(0)= aux[0] byte0 = 0 → sc6-32 = -32 → dl=-32
        //   hmask[0..16]=0 (high bit=0), qs[0..16]=0x00 (low2=0 at shift=0)
        //   q3u = 0 | (0<<2) = 0. y = -32*0 + (-4)*(-32) = 128... wait
        //   y = dl*q3u + (-4*dl) = -32*0 + (-4*(-32)) = 128
        //
        // Let me verify this explicitly:
        //   q3u=0, dl=-32, min=-4*dl=128. y = -32*0 + 128 = 128. ✓
        //
        // Alternatively: set scales_raw to encode sc6=32 for sub-block 0.
        //   When tmp=aux[2]=0, aux[0]=scales_bytes[0..4] as u32.
        //   For sc6=32 after decode:
        //     sc6(0) = (aux[0] & 0xFF) = 32 → need aux[0] byte 0 = 32 = 0x20
        //     After bit manip (tmp=0): aux[0] = (orig_aux0 & 0x0F0F0F0F) | ...
        //     So (orig_aux0 & 0xF) = 32? 32 > 15, so the lower 4 bits can't encode 32.
        //
        // The scale decoding is complex. Let me just use all-zero scales (sc6=0, dl=-32*1=-32)
        // with hmask[0..16]=0 and qs[0..16]=0x00:
        // y = -32*0 + (-4*(-32)) = 128.0
        let mut block = vec![0u8; 110];
        // hmask[0..32] = all 0 (high bit not set for any elem)
        // qs[32..96] = all 0 (low2=0 at any shift)
        // scales[96..108] = all 0 (encodes sc6=0 after bit manipulation → dl=-32)
        // d[108..110] = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        block[108..110].copy_from_slice(&d_bytes);

        let y = dequant_block(infr_core::DType::Q3K, &block).unwrap();
        assert_eq!(y.len(), 256);
        // sc6=0 → dl = 1.0*(0-32) = -32.0
        // q3u = 0 (hmask=0, qs=0), min = -4*(-32) = 128
        // y[0] = -32*0 + 128 = 128.0
        assert!(
            (y[0] - 128.0).abs() < 1e-3,
            "q3k y[0] expected 128.0, got {}",
            y[0]
        );
        // All elements should be 128.0 (same scale, q3u=0 everywhere)
        for i in 0..256 {
            assert!(
                (y[i] - 128.0).abs() < 1e-3,
                "q3k y[{i}] expected 128.0, got {}",
                y[i]
            );
        }
    }

    // ── Q4_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qs[16]]; y = d * (q4 - 8), q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_0 (ggml-quants.c l.401)
    #[test]
    fn q4_0_single_block() {
        // d = 2.0 (f16 = 0x4000), qs[0] = 0x89 (lo=9, hi=8), rest = 0x88 (lo=8, hi=8)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x89; // qs[0]: lo=9, hi=8
        for b in &mut block[3..18] {
            *b = 0x88; // lo=8, hi=8 → y = d*(8-8) = 0
        }
        let y = dequant_block(infr_core::DType::Q4_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*(9-8) = 2.0
        assert!(
            (y[0] - 2.0).abs() < 1e-5,
            "q4_0 y[0] expected 2.0, got {}",
            y[0]
        );
        // y[16] = 2.0*(8-8) = 0.0
        assert!(y[16].abs() < 1e-5, "q4_0 y[16] expected 0.0, got {}", y[16]);
        // y[1] = 2.0*(8-8) = 0.0
        assert!(y[1].abs() < 1e-5, "q4_0 y[1] expected 0.0, got {}", y[1]);
    }

    // ── Q4_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qs[16]]; y = d*q4 + m, q4 ∈ 0..15
    // Reference: llama.cpp dequantize_row_q4_1 (ggml-quants.c l.421)
    #[test]
    fn q4_1_single_block() {
        // d=1.0, m=0.5, qs[0]=0x30 (lo=0, hi=3), rest=0x00
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        block[4] = 0x30; // lo=0, hi=3
        let y = dequant_block(infr_core::DType::Q4_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 1.0*0 + 0.5 = 0.5
        assert!(
            (y[0] - 0.5).abs() < 1e-4,
            "q4_1 y[0] expected 0.5, got {}",
            y[0]
        );
        // y[16] = 1.0*3 + 0.5 = 3.5
        assert!(
            (y[16] - 3.5).abs() < 1e-4,
            "q4_1 y[16] expected 3.5, got {}",
            y[16]
        );
    }

    // ── Q5_0 ────────────────────────────────────────────────────────────────────
    // Block: [half d][uint8 qh[4]][uint8 qs[16]]; y = d*(q5 - 16), q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_0 (ggml-quants.c l.442)
    #[test]
    fn q5_0_single_block() {
        // d=1.0, qh=[0x01,0,0,0] (bit 0 → element 0 gets high bit → q5=15|16=31)
        // qs[0]=0x0F (lo=15, hi=0), rest=0
        let d_bytes = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bytes);
        block[2] = 0x01; // qh[0]: bit 0 set
        block[6] = 0x0F; // qs[0]: lo=15, hi=0
        let y = dequant_block(infr_core::DType::Q5_0, &block).unwrap();
        assert_eq!(y.len(), 32);
        // j=0: xh0 = ((1>>0)<<4)&0x10 = 16. q5 = 15|16=31. y[0] = 1.0*(31-16) = 15.0
        assert!(
            (y[0] - 15.0).abs() < 1e-5,
            "q5_0 y[0] expected 15.0, got {}",
            y[0]
        );
        // j=0: xh1 = (1>>12)&0x10 = 0. q5 = 0. y[16] = 1.0*(0-16) = -16.0
        assert!(
            (y[16] - (-16.0)).abs() < 1e-5,
            "q5_0 y[16] expected -16.0, got {}",
            y[16]
        );
    }

    // ── Q5_1 ────────────────────────────────────────────────────────────────────
    // Block: [half d][half m][uint8 qh[4]][uint8 qs[16]]; y = d*q5 + m, q5 ∈ 0..31
    // Reference: llama.cpp dequantize_row_q5_1 (ggml-quants.c l.468)
    #[test]
    fn q5_1_single_block() {
        // d=2.0, m=-1.0, qh=[0,0,0,0], qs[0]=0x1F (lo=15, hi=1)
        let d_bytes = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let m_bytes = half::f16::from_f32(-1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bytes);
        block[2..4].copy_from_slice(&m_bytes);
        // qh[4] all zero → no high bits
        block[8] = 0x1F; // qs[0]: lo=15, hi=1
        let y = dequant_block(infr_core::DType::Q5_1, &block).unwrap();
        assert_eq!(y.len(), 32);
        // y[0] = 2.0*15 + (-1.0) = 29.0
        assert!(
            (y[0] - 29.0).abs() < 1e-4,
            "q5_1 y[0] expected 29.0, got {}",
            y[0]
        );
        // y[16] = 2.0*1 + (-1.0) = 1.0
        assert!(
            (y[16] - 1.0).abs() < 1e-4,
            "q5_1 y[16] expected 1.0, got {}",
            y[16]
        );
    }
}

/// Validate that the native raw-block GPU GEMV (`linear_native`) matches the CPU dequant for each
/// affine quant type — the single upload path now that `Wt::Q` (host repack + `linear_q`) is gone.
#[cfg(test)]
mod gpu_affine_tests {
    use super::*;
    use infr_core::backend::BufferUsage;
    use infr_core::Backend;
    use infr_vulkan::VulkanBackend;

    // ── Native-block GPU-vs-CPU parity tests ────────────────────────────────
    //
    // Each test: build a known raw block, run `linear_native` GEMV with x=all-1.0,
    // compare to `dequant_unified`/`dequant_codebook` CPU sum (dot with 1.0 = weight sum).

    fn check_native(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;

        // CPU reference: sum of dequantized weights (dot with all-1.0 input)
        let (qv, sc, mn) = dequant_unified(dtype, block_bytes);
        let numel = qv.len();
        let cpu_out: f32 = (0..numel).map(|g| sc[g] * qv[g] as f32 + mn[g]).sum();

        // Upload native raw block bytes (padded to u32)
        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();

        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();

        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];

        let err = (gpu_out - cpu_out).abs();
        let rel = err / (cpu_out.abs() + 1e-6);
        assert!(
            rel < 5e-3,
            "{dtype:?} native GPU vs CPU: gpu={gpu_out} cpu={cpu_out} err={err} rel={rel}"
        );
    }

    // ── Phase 0: Q8_0 ────────────────────────────────────────────────────────

    #[test]
    fn q8_0_native_matches_cpu() {
        // d=1.5, qs: bytes 0..32 = signed values -128..127 cycling
        let d_bits = half::f16::from_f32(1.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&d_bits);
        for i in 0..32u8 {
            // values: 0,1,..,127,-128,-127,...,-97 → will cycle through positive and negative
            block[2 + i as usize] = i.wrapping_add(100); // e.g. 100,101,..,127,-128,...
        }
        check_native(infr_core::DType::Q8_0, &block);
    }

    // ── Phase 1: Q4_0, Q4_1, Q5_0, Q5_1 ─────────────────────────────────────

    #[test]
    fn q4_0_native_matches_cpu() {
        // d=2.0, qs all=0x89 (lo=9,hi=8) → mix of positive/negative after -8
        let d_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&d_bits);
        for b in &mut block[2..18] {
            *b = 0x89;
        }
        check_native(infr_core::DType::Q4_0, &block);
    }

    #[test]
    fn q4_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[4..20] {
            *b = 0x31;
        }
        check_native(infr_core::DType::Q4_1, &block);
    }

    #[test]
    fn q5_0_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&d_bits);
        // qh=0 (no high bits), qs all=0x0A → q5 values 10 (lo) and 0 (hi)
        for b in &mut block[6..22] {
            *b = 0x0A;
        }
        check_native(infr_core::DType::Q5_0, &block);
    }

    #[test]
    fn q5_1_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let m_bits = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&m_bits);
        for b in &mut block[8..24] {
            *b = 0x1F;
        }
        check_native(infr_core::DType::Q5_1, &block);
    }

    // ── Phase 2: k-quants ─────────────────────────────────────────────────────

    #[test]
    fn q2k_native_matches_cpu() {
        let mut block = vec![0u8; 84];
        block[0] = 0x03;
        block[1] = 0x03;
        for b in &mut block[16..80] {
            *b = 0x55;
        }
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q2K, &block);
    }

    #[test]
    fn q3k_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        check_native(infr_core::DType::Q3K, &block);
    }

    #[test]
    fn q4k_native_matches_cpu() {
        // d=1.0, dmin=0.5, scales[0]=0x33 → sc=3, mn=3
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[4..16]: all 0x33 → k4(0)=(3,3) for first sub-block
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        // qs: alternating 0xAB
        for b in &mut block[16..144] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q4K, &block);
    }

    #[test]
    fn q5k_native_matches_cpu() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        for b in &mut block[4..16] {
            *b = 0x33;
        }
        for b in &mut block[48..176] {
            *b = 0xAB;
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q5K block: distinct scales per sub-block + non-zero qh.
    /// The uniform tests above are insensitive to indexing bugs; this one is not.
    #[test]
    fn q5k_native_nonuniform() {
        // Build a block where each sub-block has a different scale and qh is varied.
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let dmin_bits = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&d_bits);
        block[2..4].copy_from_slice(&dmin_bits);
        // scales[0..12]: encode 8 distinct 6-bit (scale,min) pairs via k4 encoding.
        // Use simple encoding: first 4 bytes = low bits of sc (i=0..3), bytes 4..8 = low bits of mn,
        // bytes 8..12 = upper bits mixed.
        // Set them to varied values so each sub-block has a different scale.
        block[4] = 0x20; // k4(0): sc=0x20&0x3F=32, mn=block[8]&0x3F
        block[5] = 0x10; // k4(2): sc=16, mn=...
        block[6] = 0x08; // k4(4): sc computed via else branch
        block[7] = 0x04; // k4(6): sc computed via else branch
        block[8] = 0x3F; // k4(0): mn=63
        block[9] = 0x2A; // k4(2): mn=42
        block[10] = 0x15; // k4(4): (used in else branch)
        block[11] = 0x09; // k4(6): (used in else branch)
                          // block[12..16] could affect k4(4..7) upper bits; set to varied pattern
        block[12] = 0xC0; // affects k4(4): sc upper bits from (block[8]>>6)<<4 = (0x3F>>6)<<4=0
        block[13] = 0x80;
        block[14] = 0x40;
        block[15] = 0x20;
        // qh: set to varied pattern so high bits vary
        for i in 0..32usize {
            block[16 + i] = (i as u8).wrapping_mul(17).wrapping_add(1);
        }
        // qs: set to varied pattern
        for i in 0..128usize {
            block[48 + i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        check_native(infr_core::DType::Q5K, &block);
    }

    /// Non-uniform Q6K block: distinct scales per sub-block.
    #[test]
    fn q6k_native_nonuniform() {
        let d_bits = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        // ql: varied
        for i in 0..128usize {
            block[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        // qh: varied
        for i in 0..64usize {
            block[128 + i] = (i as u8).wrapping_mul(7).wrapping_add(5);
        }
        // scales: varied signed int8 values (avoid extreme negatives to keep sums finite)
        for i in 0..16usize {
            block[192 + i] = ((i as u8).wrapping_mul(5) + 8) & 0x7F;
        } // positive only
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Multi-block Q5K test: 4 blocks (in_f=1024), out_f=2. Tests cross-block access.
    #[test]
    fn q5k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        // Build 8 distinct Q5K blocks (in_f=2048, out_f=2 → weight matrix [2, 2048])
        const N_BLOCKS: usize = 8;
        const BLOCK_SZ: usize = 176;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        // Total weight elements: OUT_F * IN_F = 2 * 2048 = 4096 = 16 blocks
        const TOTAL_BLOCKS: usize = OUT_F * IN_F / NELEMS; // = OUT_F * N_BLOCKS
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        // Fill blocks with distinct, varied data
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            let dmin_bits = half::f16::from_f32(0.1).to_bits().to_le_bytes();
            w_bytes[off..off + 2].copy_from_slice(&d_bits);
            w_bytes[off + 2..off + 4].copy_from_slice(&dmin_bits);
            for i in 0..12 {
                w_bytes[off + 4 + i] = ((b * 12 + i) as u8).wrapping_mul(3) | 0x20;
            }
            for i in 0..32 {
                w_bytes[off + 16 + i] = ((b * 32 + i) as u8).wrapping_mul(17);
            }
            for i in 0..128 {
                w_bytes[off + 48 + i] = ((b * 128 + i) as u8).wrapping_mul(7).wrapping_add(3);
            }
        }
        // CPU reference: compute expected outputs using dequant_unified
        let mut cpu_outputs = [0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q5K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        // GPU: upload and run
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q5K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q5K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    /// Full-scale Q6K test matching ffn_down dimensions: out_f=1024, in_f=3072.
    #[test]
    fn q6k_native_fullscale() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = 3072;
        const OUT_F: usize = 1024;
        let n_blocks_per_row = IN_F / NELEMS; // 12
        let total_blocks = OUT_F * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BLOCK_SZ];
        for b in 0..total_blocks {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.1 + (b % 16) as f32 * 0.05)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 7 + i) as u8).wrapping_mul(11);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 3 + i) as u8).wrapping_mul(7);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        // Only check a few output elements to keep test fast
        let check_rows = [0usize, 1, 100, 1023];
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for &o in &check_rows {
            let w_row_bytes =
                &w_bytes[o * n_blocks_per_row * BLOCK_SZ..(o + 1) * n_blocks_per_row * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let cpu: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            let err = (gpu_outputs[o] - cpu).abs();
            let rel = err / (cpu.abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K fullscale out[{o}]: gpu={} cpu={cpu} err={err} rel={rel}",
                gpu_outputs[o]
            );
        }
    }

    /// Multi-block Q6K test: 8 blocks, out_f=2. Tests cross-block access.
    #[test]
    fn q6k_native_multiblock() {
        use infr_vulkan::linear::pad_to_u32_align;
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        const N_BLOCKS: usize = 4;
        const BLOCK_SZ: usize = 210;
        const NELEMS: usize = 256;
        const IN_F: usize = N_BLOCKS * NELEMS;
        const OUT_F: usize = 2;
        const TOTAL_BLOCKS: usize = OUT_F * N_BLOCKS;
        let mut w_bytes = vec![0u8; TOTAL_BLOCKS * BLOCK_SZ];
        for b in 0..TOTAL_BLOCKS {
            let off = b * BLOCK_SZ;
            let d_bits = half::f16::from_f32(0.5 + b as f32 * 0.1)
                .to_bits()
                .to_le_bytes();
            for i in 0..128 {
                w_bytes[off + i] = ((b * 128 + i) as u8).wrapping_mul(11).wrapping_add(3);
            }
            for i in 0..64 {
                w_bytes[off + 128 + i] = ((b * 64 + i) as u8).wrapping_mul(7).wrapping_add(5);
            }
            for i in 0..16 {
                w_bytes[off + 192 + i] = (((b * 16 + i) as u8).wrapping_mul(5) + 8) & 0x7F;
            }
            w_bytes[off + 208..off + 210].copy_from_slice(&d_bits);
        }
        let mut cpu_outputs = [0f32; OUT_F];
        let x: Vec<f32> = (0..IN_F).map(|i| 1.0f32 + i as f32 * 0.001f32).collect();
        for o in 0..OUT_F {
            let w_row_bytes = &w_bytes[o * N_BLOCKS * BLOCK_SZ..(o + 1) * N_BLOCKS * BLOCK_SZ];
            let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, w_row_bytes);
            let sum: f32 = (0..IN_F)
                .map(|i| (sc[i] * qv[i] as f32 + mn[i]) * x[i])
                .sum();
            cpu_outputs[o] = sum;
        }
        let padded = pad_to_u32_align(&w_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(IN_F * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(OUT_F * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            IN_F,
            OUT_F,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; OUT_F * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_outputs: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        for o in 0..OUT_F {
            let err = (gpu_outputs[o] - cpu_outputs[o]).abs();
            let rel = err / (cpu_outputs[o].abs() + 1e-3);
            assert!(
                rel < 5e-3,
                "Q6K out[{o}]: gpu={} cpu={} err={err} rel={rel}",
                gpu_outputs[o],
                cpu_outputs[o]
            );
        }
    }

    #[test]
    fn q6k_native_matches_cpu() {
        // d=0.5, scales[0..16]=0x20 (i8=32), ql=0xFF, qh=0xFF → q6=63
        let d_bits = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = +32
        block[208..210].copy_from_slice(&d_bits);
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Verify Q6K native shader handles f16 subnormal d values correctly.
    /// Real model weights use subnormal d (e.g. d_bits=0x0140 ≈ 1.9e-5), which
    /// naive f16→f32 that maps e=0 to 0 will silently zero out every output.
    #[test]
    fn q6k_native_subnormal_d() {
        // d_bits = 0x0140 (e=0, m=0x140=320): subnormal f16 ≈ 1.9073e-5
        let d_bits: u16 = 0x0140;
        let mut block = vec![0u8; 210];
        for b in &mut block[0..128] {
            *b = 0xFF;
        } // ql all-1
        for b in &mut block[128..192] {
            *b = 0xFF;
        } // qh all-1
        for b in &mut block[192..208] {
            *b = 0x20;
        } // scales = i8 +32
        block[208..210].copy_from_slice(&d_bits.to_le_bytes());
        check_native(infr_core::DType::Q6K, &block);
    }

    /// Load a real Q6K tensor from the model and verify GPU vs CPU.
    #[test]
    fn q6k_real_model_tensor() {
        use infr_vulkan::linear::pad_to_u32_align;
        let Some(model_path) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan");
                return;
            }
        };
        let g = infr_gguf::Gguf::open(&model_path).unwrap();
        // attn_v.weight blk.0: Q6K, [1024, 1024] → in_f=1024, out_f=1024
        let tensor_name = "blk.0.attn_v.weight";
        let bytes = g.tensor_bytes(tensor_name).unwrap();
        let in_f = 1024usize;
        let out_f = 1024usize;
        // CPU ref: dot each output row against x=all-1.0
        let (qv, sc, mn) = dequant_unified(infr_core::DType::Q6K, bytes);
        let numel = in_f * out_f;
        assert_eq!(qv.len(), numel, "element count mismatch");
        let x: Vec<f32> = vec![1.0f32; in_f];
        let mut cpu_out = vec![0f32; out_f];
        for o in 0..out_f {
            cpu_out[o] = (0..in_f)
                .map(|i| sc[o * in_f + i] * qv[o * in_f + i] as f32 + mn[o * in_f + i])
                .sum();
        }
        // GPU
        let padded = pad_to_u32_align(bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let xbuf = be.alloc(in_f * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(out_f * 4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            infr_core::DType::Q6K,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            in_f,
            out_f,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; out_f * 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&out_bytes).to_vec();
        let mut max_err = 0f32;
        let mut max_idx = 0;
        let mut n_zero = 0usize;
        for o in 0..out_f {
            let err = (gpu_out[o] - cpu_out[o]).abs();
            if gpu_out[o] == 0.0 && cpu_out[o].abs() > 0.1 {
                n_zero += 1;
            }
            if err > max_err {
                max_err = err;
                max_idx = o;
            }
        }
        // Print first 5 failing elements
        let mut n_print = 0;
        for o in 0..out_f {
            let rel = (gpu_out[o] - cpu_out[o]).abs() / (cpu_out[o].abs() + 1e-3);
            if rel > 5e-3 && n_print < 5 {
                eprintln!("FAIL out[{o}]: gpu={} cpu={}", gpu_out[o], cpu_out[o]);
                n_print += 1;
            }
        }
        eprintln!("Real Q6K: n_zero={n_zero}/{out_f}, max_err={max_err} at out[{max_idx}]");
        let rel = max_err / (cpu_out[max_idx].abs() + 1e-3);
        assert!(
            rel < 5e-3,
            "Real Q6K tensor: max_err={max_err} at out[{max_idx}]: gpu={} cpu={} rel={rel}",
            gpu_out[max_idx],
            cpu_out[max_idx]
        );
    }

    // ── Native-block prefill GEMM parity (matmul_native vs trusted linear_native) ──
    //
    // The tiled coopmat GEMM reuses the same per-format dqblk decode as the GEMV, so the decode is
    // already covered by the *_native_matches_cpu tests. This guards the NEW code — the 64x64 tile,
    // shared staging, and coopmat accumulation — by checking that C[m,:] from matmul_native equals
    // the GEMV linear_native(weight, A[m]) for every row m, across M spanning multiple row-tiles.
    // Weight blocks vary their f16 d per block so columns are distinguishable (catches col mixups).

    // Build one valid native block of `dtype` with f16 scale `d` and a varied payload from `seed`.
    fn native_block(dtype: infr_core::DType, d: f32, seed: u8) -> Vec<u8> {
        use infr_core::DType::*;
        let dbits = half::f16::from_f32(d).to_bits().to_le_bytes();
        match dtype {
            Q8_0 => {
                let mut b = vec![0u8; 34];
                b[0..2].copy_from_slice(&dbits);
                fill(&mut b[2..34], 17, seed);
                b
            }
            Q4K => {
                let mut b = vec![0u8; 144];
                b[0..2].copy_from_slice(&dbits); // d
                b[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes()); // dmin
                fill(&mut b[4..16], 13, seed); // 6-bit scales
                fill(&mut b[16..144], 7, seed); // qs
                b
            }
            Q6K => {
                let mut b = vec![0u8; 210];
                fill(&mut b[0..128], 7, seed); // ql
                fill(&mut b[128..192], 11, seed); // qh
                fill(&mut b[192..208], 3, seed); // i8 scales
                b[208..210].copy_from_slice(&dbits); // d
                b
            }
            other => panic!("native_block: add {other:?}"),
        }
    }

    fn check_native_gemm(dtype: infr_core::DType, m: usize) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let n = 64usize;
        let k = 256usize;
        let belems = if dtype == infr_core::DType::Q8_0 {
            32
        } else {
            256
        };
        let blocks_per_row = k / belems;

        // Weight [N, K] as native blocks (row-major). d varies per block → distinguishable columns.
        let mut wbytes: Vec<u8> = Vec::new();
        for o in 0..n {
            for bk in 0..blocks_per_row {
                let d = 0.005 * ((o % 7) as f32 + 1.0) + 0.001 * bk as f32;
                wbytes.extend_from_slice(&native_block(dtype, d, (o * 3 + bk * 5) as u8));
            }
        }
        let wbuf = be.upload_weight_bytes(&pad_to_u32_align(&wbytes)).unwrap();

        // Activations [M, K], varied per (row, col).
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.05 + ((i / k) as f32) * 0.001)
            .collect();
        let abuf = be.alloc(a.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(abuf.as_ref(), bytemuck::cast_slice(&a)).unwrap();

        // GPU GEMM → C [ceil(m/64)*64, N]. Device-local (coopmat store needs it), download via copy.
        let crows = m.div_ceil(64) * 64;
        let cbuf = be.alloc(crows * n * 4, BufferUsage::Activations).unwrap();
        let rec = be.recorder().unwrap();
        rec.matmul_native(dtype, abuf.as_ref(), wbuf.as_ref(), cbuf.as_ref(), m, k, n);
        rec.finish().unwrap();
        let mut cbytes = vec![0u8; crows * n * 4];
        be.download(cbuf.as_ref(), &mut cbytes).unwrap();
        let cgemm: &[f32] = bytemuck::cast_slice(&cbytes);

        // Reference: one GEMV per row → C[m,:]
        for row in 0..m {
            let xbuf = be.alloc(k * 4, BufferUsage::Staging).unwrap();
            be.upload(
                xbuf.as_ref(),
                bytemuck::cast_slice(&a[row * k..row * k + k]),
            )
            .unwrap();
            let ybuf = be.alloc(n * 4, BufferUsage::Readback).unwrap();
            let rec2 = be.recorder().unwrap();
            rec2.linear_native(dtype, wbuf.as_ref(), xbuf.as_ref(), ybuf.as_ref(), 1, k, n);
            rec2.finish().unwrap();
            let mut ybytes = vec![0u8; n * 4];
            be.download(ybuf.as_ref(), &mut ybytes).unwrap();
            let yref: &[f32] = bytemuck::cast_slice(&ybytes);
            // The GEMM rounds activations+weights to f16 for coopmat (GEMV keeps f32 activations), so
            // compare error against the row's largest magnitude (standard GEMM metric) — near-zero
            // outputs from cancellation otherwise blow up a pure relative error.
            let rmax = yref.iter().fold(0f32, |a, &v| a.max(v.abs()));
            for col in 0..n {
                let g = cgemm[row * n + col];
                let r = yref[col];
                let err = (g - r).abs();
                assert!(
                    err < 0.02 * rmax + 1e-4,
                    "{dtype:?} GEMM vs GEMV at [{row},{col}]: gemm={g} gemv={r} err={err} rmax={rmax}"
                );
            }
        }
    }

    #[test]
    fn q8_0_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q8_0, 70);
    }

    #[test]
    fn q4k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q4K, 70);
    }

    #[test]
    fn q6k_native_gemm_matches_gemv() {
        check_native_gemm(infr_core::DType::Q6K, 70);
    }

    // ── Native-block codebook formats (IQ4_NL/XS, MXFP4, NVFP4, TQ1_0, TQ2_0) ────
    //
    // CPU reference is `dequant_codebook` (the verified host port). GPU runs `linear_native`
    // with x=all-1.0 so the output is the sum of dequantized weights.

    fn check_native_cb(dtype: infr_core::DType, block_bytes: &[u8]) {
        let be = match VulkanBackend::new() {
            Ok(b) => b,
            Err(_) => {
                eprintln!("skip: no Vulkan GPU");
                return;
            }
        };
        use infr_vulkan::linear::pad_to_u32_align;
        let cpu = dequant_codebook(dtype, block_bytes);
        let numel = cpu.len();
        let cpu_out: f32 = cpu.iter().sum();

        let padded = pad_to_u32_align(block_bytes);
        let wbuf = be.upload_weight_bytes(&padded).unwrap();
        let x: Vec<f32> = vec![1.0f32; numel];
        let xbuf = be.alloc(x.len() * 4, BufferUsage::Staging).unwrap();
        be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        let ybuf = be.alloc(4, BufferUsage::Readback).unwrap();
        let rec = be.recorder().unwrap();
        rec.linear_native(
            dtype,
            wbuf.as_ref(),
            xbuf.as_ref(),
            ybuf.as_ref(),
            1,
            numel,
            1,
        );
        rec.finish().unwrap();
        let mut out_bytes = vec![0u8; 4];
        be.download(ybuf.as_ref(), &mut out_bytes).unwrap();
        let gpu_out: f32 = bytemuck::cast_slice(&out_bytes)[0];
        let rel = (gpu_out - cpu_out).abs() / (cpu_out.abs() + 1e-4);
        assert!(
            rel < 5e-3,
            "{dtype:?} native cb GPU vs CPU: gpu={gpu_out} cpu={cpu_out} rel={rel}"
        );
    }

    // varied non-trivial byte pattern
    fn fill(buf: &mut [u8], mul: u8, add: u8) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(mul).wrapping_add(add);
        }
    }

    #[test]
    fn iq4nl_native_matches_cpu() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.5).to_bits().to_le_bytes());
        fill(&mut block[2..18], 23, 7);
        check_native_cb(infr_core::DType::Iq4Nl, &block);
    }

    #[test]
    fn iq4xs_native_matches_cpu() {
        let mut block = vec![0u8; 136];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&0x9ce3u16.to_le_bytes()); // scales_h varied
        fill(&mut block[4..8], 53, 11); // scales_l
        fill(&mut block[8..136], 13, 3); // qs
        check_native_cb(infr_core::DType::Iq4Xs, &block);
    }

    #[test]
    fn mxfp4_native_matches_cpu() {
        let mut block = vec![0u8; 17];
        block[0] = 128; // e8m0 → d=1.0
        fill(&mut block[1..17], 29, 5);
        check_native_cb(infr_core::DType::Mxfp4, &block);
    }

    #[test]
    fn nvfp4_native_matches_cpu() {
        let mut block = vec![0u8; 36];
        block[0..4].copy_from_slice(&[0x38, 0x40, 0x48, 0x30]); // valid ue4m3 scales
        fill(&mut block[4..36], 19, 9);
        check_native_cb(infr_core::DType::Nvfp4, &block);
    }

    #[test]
    fn tq1_0_native_matches_cpu() {
        let mut block = vec![0u8; 54];
        fill(&mut block[0..52], 17, 1); // qs + qh
        block[52..54].copy_from_slice(&half::f16::from_f32(0.75).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq1_0, &block);
    }

    #[test]
    fn tq2_0_native_matches_cpu() {
        let mut block = vec![0u8; 66];
        fill(&mut block[0..64], 11, 3); // qs
        block[64..66].copy_from_slice(&half::f16::from_f32(1.25).to_bits().to_le_bytes());
        check_native_cb(infr_core::DType::Tq2_0, &block);
    }

    #[test]
    fn iq2xxs_native_matches_cpu() {
        // 2 blocks (in_f=512) to exercise cross-block + grid/sign decode.
        let mut blocks = vec![0u8; 2 * 66];
        for (bi, blk) in blocks.chunks_mut(66).enumerate() {
            blk[0..2].copy_from_slice(
                &half::f16::from_f32(1.0 + bi as f32 * 0.5)
                    .to_bits()
                    .to_le_bytes(),
            );
            fill(&mut blk[2..66], 31, (bi as u8) * 7 + 13); // qs (grid idx + signs + scale)
        }
        check_native_cb(infr_core::DType::Iq2Xxs, &blocks);
    }

    #[test]
    fn iq2xs_native_matches_cpu() {
        let mut block = vec![0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 29, 5); // qs (u16 grid idx + sign)
        fill(&mut block[66..74], 17, 1); // scales
        check_native_cb(infr_core::DType::Iq2Xs, &block);
    }

    #[test]
    fn iq2s_native_matches_cpu() {
        let mut block = vec![0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 23, 7); // qs (idx low) + sign bytes
        fill(&mut block[66..74], 13, 2); // qh
        fill(&mut block[74..82], 19, 1); // scales
        check_native_cb(infr_core::DType::Iq2S, &block);
    }

    #[test]
    fn iq3xxs_native_matches_cpu() {
        let mut block = vec![0u8; 98];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 7, 1); // qs (grid indices)
        fill(&mut block[66..98], 13, 3); // sas (scale+signs)
        check_native_cb(infr_core::DType::Iq3Xxs, &block);
    }

    #[test]
    fn iq3s_native_matches_cpu() {
        let mut block = vec![0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..66], 11, 2); // qs
        fill(&mut block[66..74], 5, 1); // qh
        fill(&mut block[74..106], 17, 3); // signs
        fill(&mut block[106..110], 3, 1); // scales
        check_native_cb(infr_core::DType::Iq3S, &block);
    }

    #[test]
    fn iq1s_native_matches_cpu() {
        let mut block = vec![0u8; 50];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        fill(&mut block[2..34], 13, 1); // qs
        fill(&mut block[34..50], 23, 7); // qh (u16: grid hi bits + scale + delta)
        check_native_cb(infr_core::DType::Iq1S, &block);
    }

    #[test]
    fn iq1m_native_matches_cpu() {
        let mut block = vec![0u8; 56];
        fill(&mut block[0..32], 17, 3); // qs
        fill(&mut block[32..48], 11, 1); // qh
                                         // scales: nonzero so packed d != 0
        block[48..56].copy_from_slice(&[0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0x3d]);
        check_native_cb(infr_core::DType::Iq1M, &block);
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    // Validate the GGUF-derived tokenizer against the HF tokenizer.json sidecar (same model).
    // Skips if the test model isn't present.
    #[test]
    fn embedded_tokenizer_matches_sidecar() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        // The sidecar tokenizer.json must sit beside the GGUF (HF cache blobs are content-addressed
        // with no sidecar, so this runs only where a snapshot ships tokenizer.json).
        let side = gguf.with_file_name("tokenizer.json");
        if !side.exists() {
            eprintln!("skip: no tokenizer.json sidecar beside the GGUF");
            return;
        }
        let g = Gguf::open(&gguf).unwrap();
        let derived = build_tokenizer(&g).unwrap();
        let sidecar = Tokenizer::from_file(&side).unwrap();
        for s in [
            "Hello world",
            "The quick brown fox.",
            "<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\n",
            "café déjà vu — 123 + 456 = 579",
            "def f(x):\n    return x * 2\n",
        ] {
            let a = derived.encode(s, false).unwrap();
            let b = sidecar.encode(s, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "token id mismatch on {s:?}");
        }
        // <think>/</think> are user-defined (non-special): skip_special must KEEP them, while real
        // special tokens (<|im_end|>) are dropped — matching the sidecar.
        let think = "<think>\nreasoning\n</think>\n\nanswer<|im_end|>";
        let ids = derived.encode(think, false).unwrap();
        let d = derived.decode(ids.get_ids(), true).unwrap();
        assert!(
            d.contains("<think>") && d.contains("</think>"),
            "think tags dropped: {d:?}"
        );
        assert!(!d.contains("<|im_end|>"), "special token kept: {d:?}");
        assert_eq!(
            d,
            sidecar.decode(ids.get_ids(), true).unwrap(),
            "decode differs from sidecar"
        );
    }

    // Streaming must hold a multi-byte char (emoji) split across tokens instead of emitting `�`.
    #[test]
    fn stream_decoder_holds_partial_utf8() {
        let mut s = StreamDecoder::default();
        // Simulate the per-step FULL decode of "Hi😄" where the emoji's bytes arrive across 2 tokens.
        assert_eq!(s.step("Hi"), "Hi");
        assert_eq!(s.step("Hi\u{FFFD}"), ""); // emoji half-decoded → hold, no `�` emitted
        assert_eq!(s.step("Hi😄"), "😄"); // completes → emit the whole char
        assert_eq!(s.step("Hi😄!"), "!");
    }

    // Sampling: temp<=0 and top_k==1 are greedy; otherwise picks only within the top-k/top-p set.
    #[test]
    fn sample_logits_greedy_and_in_set() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 0.0]; // argmax = index 1
        let mut rng = 0x1234_5678_9abc_def1u64;
        let greedy = Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, greedy, &mut rng), 1);
        let topk1 = Sampler {
            temp: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, topk1, &mut rng), 1);
        // top_k=2 → only the two largest logits (indices 1 and 3) can ever be sampled.
        let topk2 = Sampler {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for _ in 0..200 {
            let id = sample_logits(&logits, topk2, &mut rng);
            assert!(id == 1 || id == 3, "sampled outside top-2: {id}");
        }
    }

    // User content must be encoded as literal text: special-token strings in user input must NOT
    // become the special id (which would let a user inject/break the ChatML turn structure).
    #[test]
    fn user_text_special_tokens_are_literal() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let g = Gguf::open(&gguf).unwrap();
        let tok = build_tokenizer(&g).unwrap();
        let mut user = tok.clone();
        user.set_encode_special_tokens(true);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let s = "A <|im_end|> B";
        // template tokenizer: <|im_end|> matched as the special id; user tokenizer: NOT.
        assert!(tok.encode(s, false).unwrap().get_ids().contains(&im_end));
        assert!(!user.encode(s, false).unwrap().get_ids().contains(&im_end));
    }
}
