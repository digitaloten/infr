//! Vulkan-backed [`ChatModel`]s: [`Qwen35Chat`] (the agnostic batched/chunked seam, any executor)
//! and [`DenseSeamChat`] (dense/MoE on the Vulkan agnostic seam with a persistent KV session).

use super::{ChatModel, SeamBackend};
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// qwen35 (Qwen3.5) on the agnostic batched/chunked seam ([`crate::qwen35::SeamModel`]), loaded
/// ONCE on the first turn and reused after (weights stay resident across turns). One struct serves
/// every backend — Vulkan (production), CPU and Metal (reference) — and it is the same engine
/// `infr bench` times, so run and bench cannot drift apart.
pub struct Qwen35Chat {
    path: std::path::PathBuf,
    backend: SeamBackend,
    seam: Option<crate::qwen35::SeamModel>,
}

impl Qwen35Chat {
    /// Production Vulkan seam.
    pub fn new(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Vulkan)
    }

    /// Reference CPU seam (`INFR_CPU=1`).
    pub fn new_cpu(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Cpu)
    }

    /// Reference Metal seam (`INFR_METAL=1`).
    pub fn new_metal(path: std::path::PathBuf) -> Self {
        Self::with_backend(path, SeamBackend::Metal)
    }

    pub fn with_backend(path: std::path::PathBuf, backend: SeamBackend) -> Self {
        Self {
            path,
            backend,
            seam: None,
        }
    }
}

impl ChatModel for Qwen35Chat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        crate::qwen35::render_chat_messages(&self.path, messages)
    }

    fn warmup(&mut self) -> Result<()> {
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        // An undersized warmup SeamState is fine — a bigger real prompt rebuilds it (only the
        // compiled pipelines need to persist).
        let r = self.generate("Hi", 2, &mut |_| {});
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r.map(|_| ())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.seam.is_none() {
            self.seam = Some(match self.backend {
                SeamBackend::Vulkan => crate::qwen35::SeamModel::load_vulkan(&self.path)?,
                SeamBackend::Cpu => crate::qwen35::SeamModel::load_cpu(&self.path)?,
                SeamBackend::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        crate::qwen35::SeamModel::load_metal(&self.path)?
                    }
                    #[cfg(not(target_os = "macos"))]
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            });
        }
        self.seam
            .as_mut()
            .unwrap()
            .generate(prompt, max_new, |p| on_piece(p))
    }

    fn generate_constrained(
        &mut self,
        prompt: &str,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.seam.is_none() {
            self.seam = Some(match self.backend {
                SeamBackend::Vulkan => crate::qwen35::SeamModel::load_vulkan(&self.path)?,
                SeamBackend::Cpu => crate::qwen35::SeamModel::load_cpu(&self.path)?,
                SeamBackend::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        crate::qwen35::SeamModel::load_metal(&self.path)?
                    }
                    #[cfg(not(target_os = "macos"))]
                    return Err(anyhow::anyhow!(
                        "the Metal backend is only available on macOS"
                    ));
                }
            });
        }
        self.seam
            .as_mut()
            .unwrap()
            .generate_constrained(prompt, max_new, Some(constraint), |p| on_piece(p))
    }
}

/// Dense/MoE on the VULKAN agnostic seam with a persistent KV session (`INFR_SEAM=1` for
/// `infr run`): weights upload once, and every turn prefills only the token suffix that differs
/// from the previous turn — the seam twin of the bespoke `ChatSession`'s incremental prefill.
///
/// This is the default `infr run`/`infr serve` path for EVERY arch including qwen35 (Phase 3
/// cutover — see the matching comment at both CLI call sites), so it's also where MTP mode
/// (issue #33, `docs/MTP.md`) lives: `mtp_head` is `Some` once resolved+loaded, built lazily on
/// the first [`generate`](ChatModel::generate) call when [`wants_mtp`](Self::wants_mtp) is true
/// (opt-in `INFR_MTP=1`, and only for a qwen35 GGUF that actually ships an MTP head —
/// `Config::n_layer_nextn`'s doc). `INFR_MTP` unset/`0`, or a GGUF without an MTP head:
/// `wants_mtp` is always false, `mtp_head` stays `None` forever, and `generate` takes the EXACT
/// same `session` path it always has — zero risk to non-MTP models/GGUFs.
pub struct DenseSeamChat {
    model: SeamModel,
    session: Option<crate::seam_model::DenseVulkanSession>,
    mtp_head: Option<crate::mtp::MtpHeadWeights>,
    mtp_checked: bool,
}

impl DenseSeamChat {
    pub fn new(model: SeamModel) -> Self {
        Self {
            model,
            session: None,
            mtp_head: None,
            mtp_checked: false,
        }
    }

    /// MTP mode is opt-in (`INFR_MTP=1`) and Vulkan-only this phase (the invariant test + the
    /// oracle comparison in `docs/MTP.md` are both pinned on Vulkan — CPU/Metal MTP is
    /// unimplemented, not merely untested; `DenseSeamChat` IS always Vulkan, so no backend gate
    /// is needed here beyond the GGUF check). Memoized after the first call (`mtp_checked`) so a
    /// non-MTP GGUF doesn't re-parse its `Config` every turn.
    fn wants_mtp(&mut self) -> Result<bool> {
        if self.mtp_head.is_some() {
            return Ok(true);
        }
        if self.mtp_checked {
            return Ok(false);
        }
        self.mtp_checked = true;
        if std::env::var("INFR_MTP").ok().as_deref() != Some("1") {
            return Ok(false);
        }
        if self.model.config().n_layer_nextn == 0 {
            return Ok(false);
        }
        self.mtp_head = Some(crate::mtp::load_mtp_head(
            self.model.gguf(),
            self.model.config(),
        )?);
        Ok(true)
    }
}

impl ChatModel for DenseSeamChat {
    fn render(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    fn warmup(&mut self) -> Result<()> {
        let prof2 = std::env::var_os("INFR_PROF2");
        if prof2.is_some() {
            std::env::remove_var("INFR_PROF2");
        }
        let r = self.generate("Hi", 2, &mut |_| {});
        if let Some(v) = prof2 {
            std::env::set_var("INFR_PROF2", v);
        }
        r?;
        // Drop the warmup tokens so the first real prompt prefills clean slots from row 0
        // instead of forking off a garbage prefix.
        if let Some(s) = &mut self.session {
            s.reset_cache();
        }
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.wants_mtp()? {
            let head = self.mtp_head.as_ref().expect("wants_mtp loaded it");
            return crate::mtp::generate_mtp_spec_vulkan(&self.model, head, prompt, max_new, |p| {
                on_piece(p)
            });
        }
        if self.session.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.model.config().n_ctx_train);
            self.session = Some(self.model.vulkan_session(max_ctx)?);
        }
        self.model
            .generate_vulkan_session(self.session.as_mut().unwrap(), prompt, max_new, |p| {
                on_piece(p)
            })
    }

    fn generate_constrained(
        &mut self,
        prompt: &str,
        max_new: usize,
        constraint: &mut crate::grammar::Constraint,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        if self.session.is_none() {
            let max_ctx = std::env::var("INFR_MAX_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(self.model.config().n_ctx_train);
            self.session = Some(self.model.vulkan_session(max_ctx)?);
        }
        self.model.generate_vulkan_session_constrained(
            self.session.as_mut().unwrap(),
            prompt,
            max_new,
            Some(constraint),
            |p| on_piece(p),
        )
    }
}
