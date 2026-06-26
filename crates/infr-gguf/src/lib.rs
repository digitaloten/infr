//! GGUF loader — `WeightSource` impl.
//!
//! Reference: the GGUF spec and `~/Projects/llama.cpp/ggml/include/gguf.h` +
//! `conversion/diffusion_gemma.py` (authoritative tensor names + metadata keys).
//! mmap the file; parse header → metadata KVs → tensor directory; keep quant blocks as-is.
#![allow(dead_code, unused_variables)]

use infr_core::{
    error::Result,
    loader::{Metadata, TensorInfo},
    WeightSource,
};
use std::path::Path;

pub struct Gguf {
    // TODO(sonnet): memmap2::Mmap handle, parsed Metadata, Vec<TensorInfo>, data region offset.
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
}

impl Gguf {
    /// Open + parse a GGUF file (mmap, no full read into RAM).
    ///
    /// TODO(sonnet): parse magic/version, metadata KV count + entries (all GGUF value
    /// types), tensor count + directory, align to the data region. Map GGUF ggml_type ->
    /// infr_core::DType.
    pub fn open(path: &Path) -> Result<Self> {
        todo!("parse GGUF header/metadata/tensor directory")
    }
}

impl WeightSource for Gguf {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        todo!("slice the mmap'd data region for `name`")
    }
    fn chat_template(&self) -> Option<&str> {
        self.metadata.str("tokenizer.chat_template")
    }
}

#[cfg(test)]
mod tests {
    // TODO(sonnet): test against a tiny fixture GGUF (and, gated behind an env var, the real
    // DiffusionGemma Q4_K_M file) — assert metadata keys (block_count=30, context_length,
    // head_count_kv pattern) and a known tensor's shape/dtype.
}
