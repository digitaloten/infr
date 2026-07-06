//! Backward-compatible re-export shim: the chat layer (agnostic [`crate::chat::ChatModel`] trait +
//! per-backend implementations) moved to `crates/infr-llama/src/chat/` (module split — one file per
//! backend). Kept so existing `infr_llama::model::X` / `crate::model::X` call sites (`infr-cli`)
//! keep working unchanged.
pub use crate::chat::*;
