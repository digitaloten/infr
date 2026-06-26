//! Vulkan backend (`ash` + SPIR-V). The MVP `Backend` impl.
//!
//! Reference: `~/Projects/llama.cpp/ggml/src/ggml-vulkan/` and its `vulkan-shaders/*.comp`
//! (reuse the tuned quant matmul / dequant / attention shaders). Enable device features
//! `VK_KHR_cooperative_matrix`, `shaderFloat16`, `VK_KHR_16bit_storage`,
//! `VK_KHR_shader_subgroup_extended_types`. See PLAN.md.
#![allow(dead_code, unused_variables, unused_imports)]

use infr_core::{
    backend::{Buffer, BufferUsage, Capabilities, Plan},
    error::{Error, Result},
    graph::{Bindings, Graph},
    Backend,
};

/// Vulkan device + allocator + pipeline cache.
pub struct VulkanBackend {
    // TODO(sonnet): ash Entry/Instance/Device, physical device, compute queue,
    // gpu_allocator::vulkan::Allocator, descriptor pools, pipeline cache.
}

impl VulkanBackend {
    /// Initialize Vulkan: create instance, pick a GPU (prefer discrete), create a logical
    /// device + compute queue with the required extensions/features, set up the allocator.
    pub fn new() -> Result<Self> {
        // TODO(sonnet): real init. See PLAN "Dependencies & toolchain" for features.
        todo!("init Vulkan instance/device/queue/allocator")
    }
}

struct VkBuffer {
    // TODO(sonnet): vk::Buffer + gpu_allocator allocation + size.
    size: usize,
}

impl Buffer for VkBuffer {
    fn len_bytes(&self) -> usize {
        self.size
    }
}

struct VkPlan {
    // TODO(sonnet): per-op pipelines + descriptor sets + a recorded command buffer.
}

impl Plan for VkPlan {}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn capabilities(&self) -> Capabilities {
        // TODO(sonnet): query the physical device (coop-matrix, f16, max buffer range).
        todo!("query device capabilities")
    }

    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>> {
        todo!("allocate a device buffer via gpu-allocator")
    }

    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()> {
        todo!("staging upload host -> device")
    }

    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()> {
        todo!("readback device -> host")
    }

    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>> {
        todo!("lower Graph ops to SPIR-V pipelines + record command buffers")
    }

    fn execute(&self, plan: &dyn Plan, bindings: &mut Bindings) -> Result<()> {
        todo!("bind buffers, submit command buffer")
    }

    fn sync(&self) -> Result<()> {
        todo!("wait for the compute queue / fences")
    }
}
