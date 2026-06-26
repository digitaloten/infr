//! OpenAI-compatible HTTP server (axum + SSE). Talks only to `infr-engine` — never the GPU.
//!
//! Reference for the wire mapping (streaming, `reasoning_content`, tool_calls): the working
//! shim at `~/Projects/scratch/dgemma-openai-server.py`. See PLAN.md "server".
//!
//! Routes to implement:
//!   GET  /v1/models             -> { object: "list", data: [{ id, object, owned_by }] }
//!   POST /v1/chat/completions   -> chat.completion | SSE chat.completion.chunk stream
//!
//! Map `Delta::Reasoning -> delta.reasoning_content`, `Delta::Content -> delta.content`,
//! `Delta::ToolCall -> delta.tool_calls[]` (finish_reason "tool_calls").
#![allow(dead_code, unused_variables)]

use infr_engine::Engine;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessageDto>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageDto {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

/// Start the OpenAI-compatible server bound to `addr`, serving `engine`.
///
/// TODO(sonnet): build the axum `Router` (the two routes above), wrap `engine` in shared
/// state (e.g. `Arc<Mutex<Engine>>` — generation is single-stream for the MVP), implement
/// streaming via SSE, and run it on the provided tokio runtime.
pub async fn serve(engine: Engine, addr: SocketAddr) -> anyhow::Result<()> {
    todo!("build axum router + serve OpenAI endpoints")
}
