//! GPU KV caches + per-forward scratch: the MoE host KV (`MoeKv`) and the dense per-layer
//! GPU KV cache (`KvCache`), plus prefill/decode scratch buffers. Split out of `lib.rs`.
use infr_core::backend::Buffer;

/// Mixture-of-experts FFN parameters (qwen3moe): a softmax router picks `n_used` of `n_expert`
/// experts per token, each a SwiGLU FFN of inner size `n_ff_exp`, summed by renormalized top-k
/// weights (`scale` applied). Attention is identical to dense qwen3.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub scale: f32,
}

/// State for the eager MoE generation: a GPU KV cache (so context competes for VRAM, like the dense
/// path) + the streaming `ExpertPool` for `INFR_MOE_STREAM` (lazily created on first streamed layer).
pub struct MoeKv {
    pub(crate) kv: KvCache,
    pub(crate) pool: Option<infr_vulkan::ExpertPool>,
    /// Persistent decode scratch (Tier 0): the per-token activation buffers, allocated once and
    /// reused every decode step instead of created/freed per token.
    pub(crate) dec: Option<DecodeScratch>,
    /// Persistent prefill expert scratch: one reusable buffer set the grouped-expert FFN reuses for
    /// every active expert (instead of `create_buffer`/free ~8 buffers per expert per layer,
    /// ~50k/chunk). Experts serialize through it (a barrier on reuse) — which a K-sweep showed they
    /// did anyway (the win is removing the alloc churn, not concurrency). Sized for the largest chunk.
    pub(crate) pf: Option<PrefillScratch>,
    /// Record-once decode: the GPU-resident decode forward recorded into a resubmittable command
    /// buffer, keyed by its attention structure `(use_split, chunk, n_chunks)`. Replayed each token
    /// (only the params SSBO + embedding change) instead of re-recorded; re-recorded when the
    /// signature changes (every ~chunk tokens) or never if it doesn't.
    pub(crate) rec_decode: Option<((bool, usize, usize, bool), infr_vulkan::RecordedCmd)>,
}

/// A `(quants, scales, sums)` int8-activation buffer triple produced by `quant_q8`.
pub(crate) type QBufs = (Box<dyn Buffer>, Box<dyn Buffer>, Box<dyn Buffer>);

/// One reusable scratch set for a grouped prefill expert's SwiGLU. Sized for `m_pad` row capacity;
/// an expert with fewer rows uses the leading prefix.
pub(crate) struct PrefillScratch {
    pub(crate) m_pad: usize,
    pub(crate) xe: Box<dyn Buffer>,
    pub(crate) ge: Box<dyn Buffer>,
    pub(crate) ue: Box<dyn Buffer>,
    pub(crate) ae: Box<dyn Buffer>,
    pub(crate) ye: Box<dyn Buffer>,
    pub(crate) gqa: Box<dyn Buffer>,
    pub(crate) gda: Box<dyn Buffer>,
    pub(crate) gsa: Box<dyn Buffer>,
    pub(crate) dqa: Box<dyn Buffer>,
    pub(crate) dda: Box<dyn Buffer>,
    pub(crate) dsa: Box<dyn Buffer>,
}

/// Reusable GPU scratch for one decode step's forward (all buffers sized for a single token; the
/// split-K attention buffers are sized for the cache's worst-case chunk count). Held by [`MoeKv`]
/// so decode doesn't churn ~22 buffer create/free calls per token.
pub(crate) struct DecodeScratch {
    pub(crate) hidden: Box<dyn Buffer>,
    pub(crate) hn: Box<dyn Buffer>,
    pub(crate) hn2: Box<dyn Buffer>,
    pub(crate) ao: Box<dyn Buffer>,
    pub(crate) qr: Box<dyn Buffer>,
    pub(crate) kr: Box<dyn Buffer>,
    pub(crate) vr: Box<dyn Buffer>,
    pub(crate) q_f16: Box<dyn Buffer>,
    pub(crate) attn: Box<dyn Buffer>,
    pub(crate) g: Box<dyn Buffer>,
    pub(crate) u: Box<dyn Buffer>,
    pub(crate) act: Box<dyn Buffer>,
    pub(crate) y: Box<dyn Buffer>,
    pub(crate) logits: Box<dyn Buffer>,
    pub(crate) ids: Box<dyn Buffer>,
    pub(crate) wts: Box<dyn Buffer>,
    pub(crate) qa: Box<dyn Buffer>,
    pub(crate) dact: Box<dyn Buffer>,
    pub(crate) sact: Box<dyn Buffer>,
    pub(crate) pm: Box<dyn Buffer>,
    pub(crate) pl: Box<dyn Buffer>,
    pub(crate) pacc: Box<dyn Buffer>,
    /// Host-visible [pos, kv_len] u32 SSBO the `_dyn` decode kernels read, so the decode command
    /// buffer can be recorded once and replayed (only this + the embedding change per token).
    pub(crate) params: Box<dyn Buffer>,
    /// Host-visible source for this token's embedding row: the recorded buffer copies `emb_in`→`hidden`
    /// at its start, so the host just writes here (mapped, no submit) instead of a per-token upload.
    pub(crate) emb_in: Box<dyn Buffer>,
    /// lm-head scratch, folded into the replayed buffer for greedy decode (final norm + vocab GEMV +
    /// argmax → `tok`), so the whole token is one replay + a 4-byte readback.
    pub(crate) normed: Box<dyn Buffer>,
    pub(crate) final_logits: Box<dyn Buffer>,
    pub(crate) tok: Box<dyn Buffer>,
}

impl MoeKv {
    /// Tokens currently resident in the cache (the next chunk's start position).
    pub fn len(&self) -> usize {
        self.kv.len
    }
    /// True when no tokens are resident yet.
    pub fn is_empty(&self) -> bool {
        self.kv.len == 0
    }
}

/// Per-layer key/value cache held on the GPU (persists across decode steps).
pub struct KvCache {
    pub(crate) k: Vec<Box<dyn Buffer>>, // per layer: [max_ctx, n_kv*head_dim]
    pub(crate) v: Vec<Box<dyn Buffer>>,
    pub(crate) len: usize,
    pub(crate) max_ctx: usize,
    /// Record-once decode (Qwen3-style dense models): persistent per-token scratch + the recorded,
    /// replayable command buffer keyed by `(use_split, chunk, n_chunks)` — mirrors the MoE decode path.
    pub(crate) dec: Option<DenseDecodeScratch>,
    pub(crate) rec_decode: Option<((bool, usize, usize), infr_vulkan::RecordedCmd)>,
}

/// Reusable single-token decode scratch for a dense (non-MoE) Qwen3 model (allocated once, replayed).
pub(crate) struct DenseDecodeScratch {
    pub(crate) hidden: Box<dyn Buffer>,
    pub(crate) hn: Box<dyn Buffer>,
    pub(crate) qr: Box<dyn Buffer>,
    pub(crate) kr: Box<dyn Buffer>,
    pub(crate) vr: Box<dyn Buffer>,
    pub(crate) q_f16: Box<dyn Buffer>,
    pub(crate) attn: Box<dyn Buffer>,
    pub(crate) gu: Box<dyn Buffer>,
    pub(crate) act: Box<dyn Buffer>,
    pub(crate) hlast: Box<dyn Buffer>,
    pub(crate) logits: Box<dyn Buffer>,
    pub(crate) pm: Box<dyn Buffer>,
    pub(crate) pl: Box<dyn Buffer>,
    pub(crate) pacc: Box<dyn Buffer>,
    pub(crate) params: Box<dyn Buffer>,
    pub(crate) emb_in: Box<dyn Buffer>,
}

impl KvCache {
    /// Tokens currently resident in the cache (the next forward's start position).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no tokens are resident yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
