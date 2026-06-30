//! Pluggable transformer sub-blocks ("mixers"), the model-specific units the agnostic forward
//! composes. Every layer is `h += token_mix(norm(h)); h += channel_mix(norm(h))`:
//!
//! - **token mixers** (mix across positions): GQA softmax attention, gated DeltaNet linear-attn.
//! - **channel mixers** (mix across features): dense SwiGLU/GeGLU FFN, MoE expert bank.
//!
//! Each block records into a command buffer over the shared weight type ([`crate::Wt`]) + kernels
//! ([`infr_vulkan::Recorder`]); an "architecture" is just a `Config` selecting which mixer per layer.
//! Blocks are introduced incrementally and adopted by the existing forwards before the unified
//! `transformer.rs` driver replaces them.

pub(crate) mod ffn;
