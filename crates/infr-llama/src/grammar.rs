//! Grammar-constrained decoding via [`llguidance`] — the reliability tier of tool calling (llama.cpp
//! parity). A [`Constraint`] wraps an llguidance `Matcher` over the model's tokenizer and, each decode
//! step, (a) masks the logits to the grammatically-allowed tokens and (b) consumes the sampled token
//! to advance the grammar (plus any deterministically-forced "fast-forward" tokens). When tools are in
//! play the grammar forces syntactically-valid, schema-conforming `<tool_call>` JSON, so even tiny
//! models can't emit malformed calls.
//!
//! The tokenizer bridge uses [`ByteTokenizer::from_json_bytes`] (the tokenizer's serialized JSON)
//! rather than `from_tokenizer`, so it's immune to the `tokenizers` crate-version skew between infr
//! (0.20) and toktrie (0.21).
//!
//! KNOWN ISSUE (under investigation): on the live decode, `compute_mask` and `consume_token` can
//! disagree for certain tokens — the mask allows a token (e.g. `!`/`5`) that `consume_token` then
//! rejects with "forced bytes: got '{'" (surfaced as `StopReason::ParserTooComplex`, which is really
//! a generic `ParserError`). This is a toktrie canonicalization mismatch for the GGUF-derived
//! tokenizer (the unit test below, driving the same grammar over the same tokenizer, agrees — it's
//! token-dependent, exposed only by the real model's token choices). The server catches the error and
//! falls back to UNCONSTRAINED generation, so requests never fail; capable models (≥1.7B) produce
//! valid tool calls unconstrained. Fixing the bridge is what makes tiny models reliable.

use anyhow::{anyhow, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::TokEnv;
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

/// Build an llguidance [`TokEnv`] from infr's in-memory tokenizer. Serializes the tokenizer to JSON
/// and reparses it on toktrie's side (decoupling the `tokenizers` versions); `eos_ids` mark stop
/// tokens; `vocab` is the model's logit width so the token trie matches the logits exactly.
pub fn build_tok_env(tokenizer: &Tokenizer, vocab: usize, eos_ids: &[u32]) -> Result<TokEnv> {
    let json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow!("serialize tokenizer: {e}"))?;
    let mut bt = ByteTokenizer::from_json_bytes(json.as_bytes())
        .map_err(|e| anyhow!("byte tokenizer: {e}"))?;
    if !eos_ids.is_empty() {
        bt.set_eos_tokens(eos_ids);
    }
    let env = ByteTokenizerEnv::new(bt, Some(vocab)).map_err(|e| anyhow!("tok env: {e}"))?;
    Ok(env.to_env())
}

/// A live grammar constraint over a decode. Cheap-ish to construct (parser build); one per request.
pub struct Constraint {
    matcher: Matcher,
    vocab: usize,
}

impl Constraint {
    /// Construct a constraint for `grammar` over `tok_env`.
    pub fn new(tok_env: TokEnv, grammar: TopLevelGrammar) -> Result<Self> {
        let vocab = tok_env.tok_trie().vocab_size();
        let factory =
            ParserFactory::new_simple(&tok_env).map_err(|e| anyhow!("parser factory: {e}"))?;
        let factory = Arc::new(factory);
        let parser = factory
            .create_parser(grammar)
            .map_err(|e| anyhow!("create parser: {e}"))?;
        let matcher = Matcher::new(Ok(parser));
        Ok(Self { matcher, vocab })
    }

    /// Mask `logits` in place to the grammar's allowed tokens (disallowed → -inf). Then the caller
    /// samples as usual and feeds the chosen token to [`accept`](Self::accept).
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<()> {
        let mask = self.matcher.compute_mask().map_err(|e| anyhow!("{e}"))?;
        let n = logits.len().min(self.vocab);
        for (id, l) in logits.iter_mut().enumerate().take(n) {
            if !mask.is_allowed(id as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    /// The grammar's deterministically-FORCED continuation at the current state (e.g. the literal
    /// `{`/`"`/`:` bytes a JSON object must emit). These must be consumed WITHOUT sampling — the right
    /// llguidance flow is to drain forced tokens first each step, then mask+sample only a free token.
    /// Returns empty when the next token is a real choice.
    pub fn forced(&mut self) -> Vec<u32> {
        self.matcher.compute_ff_tokens()
    }

    /// Advance the grammar by one freely-sampled `token` (consume only — forced tokens are drained
    /// separately via [`forced`](Self::forced)).
    pub fn accept_one(&mut self, token: u32) -> Result<()> {
        self.matcher
            .consume_token(token)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Advance the grammar by a run of forced tokens.
    pub fn consume(&mut self, tokens: &[u32]) -> Result<()> {
        self.matcher
            .consume_tokens(tokens)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Whether the grammar has reached an accepting stop state (no further tokens required).
    pub fn stopped(&mut self) -> bool {
        self.matcher.is_stopped()
    }
}

/// Build a JSON-schema grammar constraining the tool-call BODY — a single JSON object
/// `{"name": <one-of-the-tool-names>, "arguments": <that tool's parameter schema>}` (the union over
/// the request's tools). The caller prefills the `<tool_call>` opener and constrains only this body,
/// so the grammar stays pure JSON over normal byte tokens (no special-token / byte-grammar mismatch).
/// Used for `tool_choice: "required"` / a named tool — the model MUST emit one valid, schema-conforming
/// call. `tools` is the OpenAI `tools` array.
pub fn forced_tool_call_grammar(tools: &Value) -> Result<TopLevelGrammar> {
    let arr = tools
        .as_array()
        .ok_or_else(|| anyhow!("`tools` is not an array"))?;
    // One JSON-schema alternative per tool: { name: const, arguments: <params> }.
    let mut alts: Vec<Value> = Vec::new();
    for t in arr {
        let f = t
            .get("function")
            .ok_or_else(|| anyhow!("tool missing `function`"))?;
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool missing `function.name`"))?;
        let params = f
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));
        alts.push(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "const": name },
                "arguments": params,
            },
            "required": ["name", "arguments"],
            "additionalProperties": false,
        }));
    }
    let call_schema = serde_json::json!({ "anyOf": alts });
    Ok(TopLevelGrammar::from_json_schema(call_schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn qwen3_06b() -> Option<PathBuf> {
        let base =
            dirs_home()?.join(".cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
        std::fs::read_dir(&base)
            .ok()?
            .filter_map(|e| e.ok())
            .find_map(|e| {
                let p = e.path().join("Qwen3-0.6B-Q4_K_M.gguf");
                p.exists().then_some(p)
            })
    }
    fn dirs_home() -> Option<PathBuf> {
        std::env::var_os("HOME").map(PathBuf::from)
    }

    /// End-to-end: build the TokEnv from a real GGUF tokenizer, build the forced tool-call grammar,
    /// and verify the constraint actually restricts the first token to a strict subset of the vocab
    /// (the grammar must START a `<tool_call>` — not "anything goes"). Self-skips without the model.
    #[test]
    fn forced_grammar_constrains_first_token() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");

        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);
        let grammar = forced_tool_call_grammar(&tools).expect("grammar");
        let mut c = Constraint::new(env, grammar).expect("constraint");

        // Drive the constraint, greedily picking the first allowed token each step. (This naive picker
        // wanders inside JSON strings and may never choose a closing quote, so we don't require full
        // termination — only that every step stays constrained and `accept` never rejects, i.e. the
        // mask and the parser agree. Full termination is covered by the live-model server test.)
        // Mirror the real decode loop: drain forced tokens first, else mask + argmax-pick a free token.
        let mut out: Vec<u32> = Vec::new();
        for step in 0..60 {
            if c.stopped() {
                break;
            }
            let forced = c.forced();
            if !forced.is_empty() {
                c.consume(&forced).expect("consume forced");
                out.extend(forced);
                continue;
            }
            let mut logits = vec![0.0f32; cfg.vocab];
            c.apply_mask(&mut logits).expect("mask");
            let allowed = logits.iter().filter(|l| l.is_finite()).count();
            assert!(allowed > 0, "step {step}: grammar masked out EVERY token");
            assert!(
                allowed < cfg.vocab,
                "step {step}: grammar allowed everything"
            );
            let tok = crate::sampling::argmax(&logits) as u32;
            // The crux: argmax over the mask must yield a token `accept_one` accepts (mask/consume
            // agree — no special-token/byte or canonicalization mismatch).
            c.accept_one(tok).expect("accept_one must agree with mask");
            out.push(tok);
        }
        let text = tok.decode(&out, false).expect("decode");
        eprintln!("constrained prefix: {text:?}");
        // The constrained body is JSON — it must begin a `{"name": ...}` object.
        let t = text.trim_start();
        assert!(t.starts_with('{'), "expected a JSON object, got {text:?}");
        assert!(
            t.contains("name"),
            "JSON should constrain toward the `name` key: {text:?}"
        );
    }
}
