//! Model acquisition: resolve a reference to a local GGUF, reusing/extending the **Ollama
//! store** (same dir + format) so existing Ollama downloads work with zero re-download.
//!
//! See PLAN.md "fetch / model acquisition". Store layout:
//!   $OLLAMA_MODELS | ~/.ollama/models  (override with $INFR_MODELS)
//!     manifests/<registry>/<ns>/<name>/<tag>   (OCI-style JSON)
//!     blobs/sha256-<digest>                    (layers; model layer == the GGUF)
#![allow(dead_code, unused_variables)]

use infr_core::error::{Error, Result};
use std::path::PathBuf;

/// A parsed model reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRef {
    /// `hf:org/repo[:file.gguf]`
    Hf { repo: String, file: Option<String> },
    /// `ollama:name[:tag]` (tag defaults to `latest`)
    Ollama { name: String, tag: String },
    /// A plain filesystem path to a `.gguf`.
    Path(PathBuf),
}

impl ModelRef {
    /// Parse `hf:…`, `ollama:…`, or a filesystem path.
    ///
    /// TODO(sonnet): implement + unit-test the grammar (incl. default `:latest` for ollama,
    /// optional `:file` for hf, and bare paths).
    pub fn parse(s: &str) -> Result<Self> {
        todo!("parse model ref grammar")
    }
}

/// The shared on-disk model store (Ollama-compatible).
pub struct Store {
    pub root: PathBuf,
}

impl Store {
    /// Locate the store: `$INFR_MODELS`, else `$OLLAMA_MODELS`, else `~/.ollama/models`.
    pub fn discover() -> Result<Self> {
        todo!("resolve store root from env / home")
    }

    /// If the referenced model already exists locally, return the GGUF blob path
    /// (read the manifest, find the `application/vnd.ollama.image.model` layer).
    ///
    /// TODO(sonnet): parse an Ollama manifest + map digest -> blobs/sha256-<digest>.
    pub fn resolve(&self, r: &ModelRef) -> Result<Option<PathBuf>> {
        todo!("manifest lookup -> gguf blob path")
    }
}

/// Download a model into the store (Ollama registry pull, or HF hub), returning the GGUF
/// path. Writes blobs + manifest in Ollama format so `ollama` sees it too.
///
/// TODO(sonnet): implement HF (`resolve/main/...`, `HF_TOKEN`) and Ollama registry pull,
/// with progress + checksum verification.
pub fn pull(r: &ModelRef) -> Result<PathBuf> {
    todo!("download into the shared store")
}

/// Resolve from the store if present, otherwise pull. Used by `infr run` / `infr serve`.
pub fn ensure(r: &ModelRef) -> Result<PathBuf> {
    if let ModelRef::Path(p) = r {
        return Ok(p.clone());
    }
    let store = Store::discover()?;
    if let Some(p) = store.resolve(r)? {
        return Ok(p);
    }
    pull(r)
}
