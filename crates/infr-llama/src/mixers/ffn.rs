//! Dense SwiGLU channel-mixer: `out = (silu(gate(x)) * up(x)) · downᵀ`.
//!
//! Two weight layouts share one block: `Fused` is the dense `Llama` path's `[2*n_ff, n_embd]`
//! gate‖up (one GEMV → `silu_mul_fused`); `Split` is separate gate/up weights (two GEMVs →
//! `silu_mul`), as qwen35 carries them. Pure recording — the caller owns buffer allocation and
//! scratch lifetimes, so both the per-op (qwen35) and resident (Llama) styles can use it.

use crate::{rec_linear, rec_linear_add, Wt};
use infr_core::backend::Buffer;
use infr_vulkan::Recorder;

/// gate‖up weight layout for the SwiGLU block.
pub(crate) enum GateUp<'a> {
    /// Fused `[2*n_ff, n_embd]` (gate rows then up rows): one GEMV + `silu_mul_fused`.
    #[allow(dead_code)] // the fused-gu shape (see combined_gu); qwen35 adopts it next
    Fused(&'a Wt),
    /// Separate gate/up `[n_ff, n_embd]`: two GEMVs + `silu_mul`.
    Split { gate: &'a Wt, up: &'a Wt },
}

/// Record the SwiGLU sequence. `xn` is the already-RMSNormed input `[rows, n_embd]`; `act` is
/// `[rows, n_ff]` scratch; `out` is `[rows, n_embd]`. For `Fused`, `gu` is `[rows, 2*n_ff]` scratch
/// and `g`/`u` are unused; for `Split`, `g`/`u` are `[rows, n_ff]` scratch and `gu` is unused.
///
/// `residual`: when `Some(res)` the down projection fuses the residual add (`out = act·downᵀ + res`,
/// the dense `Llama` decode path, typically `out == res == hidden` for in-place); `None` writes the
/// raw projection (qwen35 adds the residual itself).
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_swiglu(
    rec: &Recorder,
    xn: &dyn Buffer,
    gate_up: GateUp,
    down: &Wt,
    gu: &dyn Buffer,
    g: &dyn Buffer,
    u: &dyn Buffer,
    act: &dyn Buffer,
    out: &dyn Buffer,
    residual: Option<&dyn Buffer>,
    rows: usize,
    n_embd: usize,
    n_ff: usize,
) {
    match gate_up {
        GateUp::Fused(w) => {
            rec_linear(rec, w, xn, gu, rows, n_embd, 2 * n_ff);
            rec.silu_mul_fused(gu, act, rows, n_ff);
        }
        GateUp::Split { gate, up } => {
            rec_linear(rec, gate, xn, g, rows, n_embd, n_ff);
            rec_linear(rec, up, xn, u, rows, n_embd, n_ff);
            rec.silu_mul(g, u, act, rows * n_ff);
        }
    }
    match residual {
        Some(res) => rec_linear_add(rec, down, act, res, out, rows, n_ff, n_embd),
        None => rec_linear(rec, down, act, out, rows, n_ff, n_embd),
    }
}
