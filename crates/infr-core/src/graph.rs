//! Backend-agnostic compute IR.
//!
//! The model layer builds a [`Graph`] — an explicit, ordered list of semantic [`Op`]s over
//! typed [`TensorId`] handles — and a [`crate::backend::Backend`] compiles + executes it
//! however it likes (Vulkan SPIR-V, CPU loops, CUDA, ROCm, Metal, MLX). See PLAN.md
//! "The backend abstraction".
//!
//! ## Why an op-list, not a pure DAG
//!
//! The real transformer forward is imperative: it reuses scratch buffers, RoPEs in place, and
//! writes K/V into a persistent cache at a running offset. A pure SSA DAG can't express those
//! aliasing/stateful writes cleanly, so [`Graph`] is an **ordered list** of ops, each naming the
//! tensor handles it reads and the handle it writes (`dst`). Two ops may legally write the same
//! handle (in-place / scratch reuse) — order is significant, exactly like a command buffer.
//!
//! ## Composite ops
//!
//! Ops are *composite/semantic* (e.g. [`Op::Attention`], [`Op::QkNorm`]) rather than scalar
//! primitives, so a GPU backend can map each one straight to a hand-fused kernel (no perf loss)
//! while a CPU backend runs a plain loop. A future backend may either implement the composites
//! directly or add a lowering pass that decomposes them into primitives.

use crate::tensor::{TensorDesc, TensorId};

/// Attention masking mode. SWA layers (Gemma) mask beyond a sliding window; the rest are causal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    /// Causal full attention (every position attends to all earlier positions).
    Causal,
    /// Causal sliding-window attention with the given window size (in tokens).
    SlidingWindow(usize),
}

/// Activation used by the gated FFN (`act(gate) * up`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// SwiGLU: `silu(gate) * up` (Llama / Qwen).
    Silu,
    /// GeGLU: `gelu_tanh(gate) * up` (Gemma).
    Gelu,
}

/// How a tensor handle is provisioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    /// Per-step input bound at execute time via [`Bindings`] (e.g. the embedded hidden state,
    /// position ids, the KV cache). The backend does NOT allocate these.
    Input,
    /// Model weight bound from the loader via [`Bindings`]. Read-only.
    Weight,
    /// Backend-allocated scratch / activation, lives for the duration of one execute.
    Internal,
    /// An [`Internal`](TensorKind::Internal) tensor whose final value is read back by the caller
    /// (collected into [`Bindings::outputs`] after execute).
    Output,
}

/// Declaration of a tensor handle: its shape/dtype and how it's provisioned.
#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub desc: TensorDesc,
    pub kind: TensorKind,
    /// Optional debug label (op/tensor name) for profiling + error messages.
    pub label: Option<String>,
}

/// Semantic ops. Each names the handles it reads plus the `dst` it writes. Grow as models need.
///
/// Dimensions that aren't derivable from the operand descs are carried inline (e.g. `n_head`,
/// `head_dim`) so a backend can execute an op without re-deriving layout from shapes.
#[derive(Clone, Debug)]
pub enum Op {
    /// `dst = rmsnorm(x) * weight`, normalizing over the last `dim` of each of `rows` rows.
    /// A weightless RMSNorm (Gemma V-norm) sets `weight` to a ones tensor.
    RmsNorm {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// `dst[m, out_f] = x[m, in_f] · weightᵀ`. `weight` may be any (quantized) dtype; the backend
    /// dispatches the kernel (GEMV/GEMM/MMQ on GPU, dequant+matvec on CPU).
    Linear {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        m: u32,
        in_f: u32,
        out_f: u32,
    },
    /// Per-head RMSNorm of `x` (`rows × n_head × head_dim`) with a per-`head_dim` `weight`
    /// (Qwen3 / Gemma Q-norm and K-norm). In place when `dst == x`.
    QkNorm {
        x: TensorId,
        weight: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    },
    /// NEOX RoPE over the first `rope_dim` of each head. `positions` is an i32 tensor of length
    /// `rows`. `freq_factors`, if present, divides per-pair angles (Gemma proportional RoPE).
    Rope {
        x: TensorId,
        positions: TensorId,
        dst: TensorId,
        rows: u32,
        n_head: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        freq_factors: Option<TensorId>,
    },
    /// Append `src` (`rows × row_stride`) into the persistent KV `cache` starting at row `pos`,
    /// casting to the cache dtype (typically f16). Stateful write — order matters.
    WriteKv {
        src: TensorId,
        cache: TensorId,
        rows: u32,
        row_stride: u32,
        pos: u32,
    },
    /// Scaled-dot-product attention. `q` is `rows × n_head × head_dim`; `k_cache`/`v_cache` hold
    /// `kv_len` rows of `n_kv × head_dim`. GQA when `n_head > n_kv`. `dst` is `rows × n_head ×
    /// head_dim`. `pos` is the absolute position of the first query row (for masking).
    Attention {
        q: TensorId,
        k_cache: TensorId,
        v_cache: TensorId,
        dst: TensorId,
        rows: u32,
        kv_len: u32,
        n_head: u32,
        n_kv: u32,
        head_dim: u32,
        scale: f32,
        mask: AttnMask,
        pos: u32,
    },
    /// Gated FFN activation over a fused `gate||up` buffer (`rows × 2*nff`): `dst = act(gate)*up`
    /// (`rows × nff`). `up_off` shifts the `up` read by a whole-element offset (Gemma per-layer
    /// embedding consumes a layer-major slice of a bigger buffer); 0 for the normal fused case.
    GatedAct {
        gate_up: TensorId,
        dst: TensorId,
        rows: u32,
        nff: u32,
        act: Activation,
        up_off: u32,
    },
    /// `dst[i] = a[i] + b[i]` (residual add). In place when `dst == a`.
    Add {
        a: TensorId,
        b: TensorId,
        dst: TensorId,
        n: u32,
    },
    /// `dst[i] = x[i] * s` (Gemma per-layer output scale, embedding scale).
    Scale {
        x: TensorId,
        dst: TensorId,
        s: f32,
        n: u32,
    },
    /// `dst[i] = cap * tanh(x[i] / cap)` (Gemma final-logit softcap).
    Softcap {
        x: TensorId,
        dst: TensorId,
        cap: f32,
        n: u32,
    },
    /// Copy `n` elements `src[src_off..] -> dst[dst_off..]` (extract last row, gather a slice).
    Copy {
        src: TensorId,
        src_off: u32,
        dst: TensorId,
        dst_off: u32,
        n: u32,
    },
}

/// An ordered op-list over declared tensor handles. Node index in `tensors` == [`TensorId`].
#[derive(Default)]
pub struct Graph {
    pub tensors: Vec<TensorDecl>,
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorId>,
    pub weights: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn decl(&mut self, desc: TensorDesc, kind: TensorKind) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(TensorDecl {
            desc,
            kind,
            label: None,
        });
        id
    }

    /// Declare a per-step input (bound at execute time).
    pub fn input(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Input);
        self.inputs.push(id);
        id
    }

    /// Declare a model weight (bound from the loader).
    pub fn weight(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Weight);
        self.weights.push(id);
        id
    }

    /// Declare backend-allocated scratch.
    pub fn internal(&mut self, desc: TensorDesc) -> TensorId {
        self.decl(desc, TensorKind::Internal)
    }

    /// Declare a read-back output.
    pub fn output(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.decl(desc, TensorKind::Output);
        self.outputs.push(id);
        id
    }

    /// Attach a debug label to a tensor handle.
    pub fn label(&mut self, id: TensorId, label: impl Into<String>) -> TensorId {
        self.tensors[id.0 as usize].label = Some(label.into());
        id
    }

    /// Append an op to the list.
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn desc(&self, id: TensorId) -> &TensorDesc {
        &self.tensors[id.0 as usize].desc
    }

    pub fn kind(&self, id: TensorId) -> TensorKind {
        self.tensors[id.0 as usize].kind
    }
}
