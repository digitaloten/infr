//! The GPU backend seam — the ONLY GPU-aware trait. Everything above is generic over it.
//!
//! Object-safe on purpose so the engine can hold `Arc<dyn Backend>` and stay blind to
//! whether Vulkan / CUDA / ROCm / Metal is underneath. See PLAN.md "backend abstraction".

use crate::error::Result;
use crate::graph::{Bindings, Graph};

/// Device capabilities the graph compiler queries to pick fast vs fallback kernels.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub name: String,
    pub f16: bool,
    pub cooperative_matrix: bool,
    pub max_buffer_bytes: u64,
    pub unified_memory: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    Weights,
    Activations,
    Staging,
}

/// Opaque device-memory handle owned by a backend.
pub trait Buffer: Send + Sync {
    fn len_bytes(&self) -> usize;
}

/// A compiled, ready-to-run graph (pipelines + command buffers for Vulkan, etc.).
pub trait Plan: Send + Sync {}

/// A compute device. Implementations: `infr-vulkan` (MVP), later CUDA / ROCm / Metal.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;

    // ---- memory ----
    fn alloc(&self, bytes: usize, usage: BufferUsage) -> Result<Box<dyn Buffer>>;
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> Result<()>;
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> Result<()>;

    // ---- execution (compile once, execute per token/step) ----
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Plan>>;
    fn execute(&self, plan: &dyn Plan, bindings: &mut Bindings) -> Result<()>;
    fn sync(&self) -> Result<()>;
}
