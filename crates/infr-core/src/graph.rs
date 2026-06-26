//! Backend-agnostic compute graph.
//!
//! The model layer builds a `Graph` of semantic [`Op`]s over [`TensorId`] handles; a
//! [`crate::backend::Backend`] compiles + executes it however it likes (Vulkan SPIR-V,
//! CUDA, ROCm, …). See PLAN.md "The backend abstraction".

use crate::tensor::{TensorDesc, TensorId};
use std::collections::HashMap;

/// Attention masking mode (DiffusionGemma mixes sliding-window and full-attention layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    Full,
    /// Sliding-window attention with the given window size (in tokens).
    SlidingWindow(usize),
}

/// Semantic tensor ops. Grow this set as the model needs more.
#[derive(Clone, Debug)]
pub enum Op {
    /// External input, bound at execute time via [`Bindings`].
    Input,
    /// Model weight, bound from the loader via [`Bindings`].
    Weight,
    MatMul {
        a: TensorId,
        b: TensorId,
    },
    Dequant {
        src: TensorId,
    },
    RmsNorm {
        x: TensorId,
        weight: TensorId,
        eps: f32,
    },
    Rope {
        x: TensorId,
        positions: TensorId,
        theta: f32,
    },
    Attention {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        mask: AttnMask,
    },
    /// Mixture-of-experts feed-forward (top-k routed).
    MoeFfn {
        x: TensorId,
        router: TensorId,
        gate: TensorId,
        up: TensorId,
        down: TensorId,
        active_k: u32,
    },
    Softmax {
        x: TensorId,
    },
    Add {
        a: TensorId,
        b: TensorId,
    },
    Mul {
        a: TensorId,
        b: TensorId,
    },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub desc: TensorDesc,
}

/// A DAG of ops. Node index == [`TensorId`].
#[derive(Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorId>,
    pub weights: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, op: Op, desc: TensorDesc) -> TensorId {
        let id = TensorId(self.nodes.len() as u32);
        self.nodes.push(Node { op, desc });
        id
    }

    pub fn input(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.push(Op::Input, desc);
        self.inputs.push(id);
        id
    }

    pub fn weight(&mut self, desc: TensorDesc) -> TensorId {
        let id = self.push(Op::Weight, desc);
        self.weights.push(id);
        id
    }

    pub fn op(&mut self, op: Op, desc: TensorDesc) -> TensorId {
        self.push(op, desc)
    }

    pub fn mark_output(&mut self, id: TensorId) {
        self.outputs.push(id);
    }

    pub fn desc(&self, id: TensorId) -> &TensorDesc {
        &self.nodes[id.0 as usize].desc
    }
}

/// Binds graph `Input`/`Weight` ids to backend buffers and collects `Output` buffers.
///
/// TODO(sonnet): finalize the binding API once the Vulkan backend lands. For now this is
/// the seam the engine uses to feed inputs/weights and read outputs without knowing the
/// concrete buffer type.
#[derive(Default)]
pub struct Bindings {
    /// Opaque per-id binding keys the backend understands (e.g. allocated buffer indices).
    pub bound: HashMap<TensorId, u64>,
}
