//! Model acquisition: resolve a reference to a local GGUF, pulling from HuggingFace or the Ollama
//! registry over plain HTTP (no external CLI). Downloads land in **our own** content-addressed
//! store (never the system Ollama dirs), with resume + a progress bar.
//!
//! See PLAN.md §"fetch / model acquisition (infr-hub)". Store layout (root = `$INFR_MODELS` or
//! `$XDG_CACHE_HOME/infr/models`):
//!
//! ```text
//!   manifests/registry.ollama.ai/<ns>/<name>/<tag>   (ollama pulls)
//!   manifests/huggingface.co/<org>/<repo>/<file>     (hf pulls)
//!   blobs/sha256-<digest>                            (layer blobs; model layer == GGUF)
//! ```

mod model_ref;
mod pull;
mod store;

pub use model_ref::ModelRef;
pub use pull::pull;
pub use store::Store;

use infr_core::error::Result;
use std::path::PathBuf;

/// Resolve from the store if present, otherwise pull.  Used by `infr run` / `infr serve`.
///
/// - `Path(p)` → returned immediately.
/// - Everything else → [`Store::discover`] + [`Store::resolve`]; if not cached → [`pull`].
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
