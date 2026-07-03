//! GPU weight-storage types: a projection weight (f16 / native-quant blocks), MoE expert
//! weights (GPU-resident or host-backed), and a layer FFN (dense gate||up + down, or a MoE
//! bank). Mechanically split out of `lib.rs` (no logic change).
use crate::{dequant_block, f32_to_f16_sat};
use anyhow::{anyhow, Result};
use infr_core::backend::Buffer;
use infr_core::WeightSource;
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;

/// A projection weight on the GPU: f16, unified repacked quant, or native raw-block quant.
///
/// - `F16`: f16 weight buffer (float or codebook-quant host-dequanted → f16)
/// - `Q`: unified repacked affine quant (q/s/m buffers, `dq = s·u8 + m`); fallback when native is
///   disabled (`INFR_NONATIVE=1`) or for grid/codebook quants under `INFR_NATIVE=1`.
/// - `Native`: raw GGUF block bytes, padded to u32 alignment, dequantized in-shader (decode-once
///   GEMV + tiled coopmat GEMM). The DEFAULT for optimized affine quants — faster decode + prefill
///   and smaller VRAM (see [`is_native_default`]); `INFR_NATIVE=1` extends it to all formats.
pub(crate) enum Wt {
    F16(Box<dyn Buffer>),
    /// Raw native-block bytes on the GPU; `dtype` identifies the dequant shader.
    Native {
        buf: Box<dyn Buffer>,
        dtype: infr_core::DType,
    },
}
impl Wt {}

/// Upload a projection weight, keeping quantized weights quantized in-VRAM (else convert to f16).
///
/// - Affine quants (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0…) → `Wt::Native` (raw block bytes, in-shader
///   decode-once dequant — faster decode + prefill, smaller VRAM). These have the `native_id_*`
///   decode GEMV shaders ([`is_native_default`]).
/// - Codebook quants (IQ*/TQ*/fp4) and float types (F16/F32/BF16) → host dequant → f16 → `Wt::F16`.
///   The i-quants have no decode-GEMV shader yet, so they stay on f16 until those land.
pub(crate) fn upload_wt(be: &VulkanBackend, g: &Gguf, name: &str) -> Result<Wt> {
    let dtype = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .dtype;
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    upload_wt_bytes(be, dtype, bytes)
}

/// Like [`upload_wt`] but from a raw byte slice + dtype — lets a stacked MoE expert tensor be sliced
/// per expert (each expert is a contiguous block of the `*_exps` tensor) and uploaded individually.
pub(crate) fn upload_wt_bytes(
    be: &VulkanBackend,
    dtype: infr_core::DType,
    bytes: &[u8],
) -> Result<Wt> {
    // Native-block path: raw upload + in-shader dequant — for every quant format with the dense
    // native pipeline (decode GEMV + prefill GEMM; see `native_dense_supported`). Only float types
    // (F16/F32/BF16, not quants) fall to the host dequant → f16 path.
    if infr_vulkan::linear::native_dense_supported(dtype) {
        let padded = infr_vulkan::linear::pad_to_u32_align(bytes);
        return Ok(Wt::Native {
            buf: be
                .upload_weight_bytes(&padded)
                .map_err(|e| anyhow!("native upload: {e}"))?,
            dtype,
        });
    }
    // Float types → host dequant to f32 → f16.
    let f16_bytes: Vec<u8> = dequant_block(dtype, bytes)?
        .iter()
        .flat_map(|&v| f32_to_f16_sat(v).to_bits().to_le_bytes())
        .collect();
    Ok(Wt::F16(be.upload_weight_bytes(&f16_bytes)?))
}

pub(crate) fn rec_linear(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::F16(b) => rec.linear(b.as_ref(), x, y, rows, in_f, out_f),
        Wt::Native { buf, dtype } => {
            rec.linear_native(*dtype, buf.as_ref(), x, y, rows, in_f, out_f)
        }
    }
}

/// `y = x·Wᵀ + residual` (fused-residual GEMV), dispatching on how `W` is stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rec_linear_add(
    rec: &infr_vulkan::Recorder,
    w: &Wt,
    x: &dyn Buffer,
    residual: &dyn Buffer,
    y: &dyn Buffer,
    rows: usize,
    in_f: usize,
    out_f: usize,
) {
    match w {
        Wt::F16(b) => rec.linear_add(b.as_ref(), x, residual, y, rows, in_f, out_f),
        Wt::Native { buf, dtype } => {
            rec.linear_add_native(*dtype, buf.as_ref(), x, residual, y, rows, in_f, out_f)
        }
    }
}
/// VRAM the model's weights will occupy once resident, split dense vs MoE-expert. Experts are
/// tracked separately so a future expert-streaming / partial-offload mode can budget them apart
/// from the always-resident dense weights — for a dense model `expert` is 0.
#[derive(Clone, Copy, Debug)]
pub struct WeightFootprint {
    /// Always-resident weights: projections, embeddings, norms.
    pub dense: u64,
    /// MoE expert weights (GGUF `*_exps` stacked tensors). 0 for dense models.
    pub expert: u64,
}
impl WeightFootprint {
    /// All-resident footprint: dense + every expert kept in VRAM.
    pub fn total(&self) -> u64 {
        self.dense + self.expert
    }

    /// Footprint if experts are STREAMED through an `n_slots`-slot pool of `stride`-byte slots
    /// (a VRAM slot pool) instead of all kept resident: `dense + n_slots·stride`, bounded
    /// regardless of the model's expert count. The MoE loader picks all-resident ([`total`]) when it
    /// fits VRAM, else reserves this and streams. (`stride` = one expert's max packed weight bytes.)
    pub fn streaming_total(&self, n_slots: usize, stride: usize) -> u64 {
        self.dense + n_slots as u64 * stride as u64
    }
}

/// Resident VRAM bytes for one tensor, mirroring [`upload_wt`]'s path so the estimate matches what
/// actually gets allocated: native raw blocks (padded to u32) for every quant format, else f16
/// (float/norms dequanted to half).
pub(crate) fn tensor_resident_bytes(dtype: infr_core::DType, numel: usize, nbytes: usize) -> u64 {
    if infr_vulkan::linear::native_dense_supported(dtype) {
        ((nbytes + 3) & !3) as u64 // raw blocks, padded to u32 alignment
    } else {
        (numel * 2) as u64 // f16
    }
}

/// Sum the resident weight footprint across all tensors (MoE-aware). Enumerating every tensor means
/// stacked expert tensors are counted in full, so this is correct for MoE the moment the arch is
/// supported. `token_embd` is excluded (it lives in host RAM for the CPU embedding gather) unless
/// the lm head is tied to it (no `output.weight`), where an f16 copy is uploaded to VRAM.
pub fn weight_footprint(g: &Gguf) -> WeightFootprint {
    let has_output = g.tensors().iter().any(|t| t.name == "output.weight");
    let mut dense = 0u64;
    let mut expert = 0u64;
    for t in g.tensors() {
        let numel: usize = t.shape.iter().product();
        if t.name == "token_embd.weight" {
            if !has_output {
                dense += (numel * 2) as u64; // tied lm head, uploaded as f16
            }
            continue;
        }
        let bytes = tensor_resident_bytes(t.dtype, numel, t.nbytes);
        if t.name.contains("_exps") {
            expert += bytes;
        } else {
            dense += bytes;
        }
    }
    WeightFootprint { dense, expert }
}
