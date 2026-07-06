//! Shared model-config types. (The bespoke KV-cache/scratch structs that lived here died with
//! the bespoke engine — the seam holds its state in `seam::SeamKv`, including qwen35's
//! gated-DeltaNet conv/S state.)

/// Routed-expert (MoE) shape: expert count, top-k, per-expert FFN width, routed-weight scale.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
    pub scale: f32,
}
