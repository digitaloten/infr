//! Milestone 1 smoke test (runnable on real hardware, not a unit test):
//! initialize the Vulkan backend and run an f16 cooperative-matrix matmul on the GPU,
//! then compare against a CPU reference within tolerance.
//!
//! Run with: `cargo run -p infr-vulkan --example smoke`
//!
//! TODO(sonnet): implement once `VulkanBackend` + a matmul pipeline exist.

fn main() -> infr_core::Result<()> {
    todo!("VulkanBackend::new(); upload A,B; dispatch coop-matrix matmul; download; assert ~== CPU")
}
