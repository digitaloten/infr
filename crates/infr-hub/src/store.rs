//! infr's own on-disk model store. The blob/manifest layout is OCI/Ollama-style (content-addressed
//! blobs + small JSON manifests, so registry pulls dedup naturally), but the ROOT is our own cache
//! dir — we never read or write the system Ollama dirs (`~/.ollama`, `/var/lib/ollama`).
//!
//! Layout inside `root` (default `$XDG_CACHE_HOME/infr/models`):
//! ```text
//! manifests/registry.ollama.ai/<ns>/<name>/<tag>   (ollama pulls)
//! manifests/huggingface.co/<org>/<repo>/<file>     (hf pulls)
//! blobs/sha256-<hex>                               (layer blobs; model layer == GGUF)
//! ```

use crate::model_ref::ModelRef;
use infr_core::error::{Error, Result};
use serde::Deserialize;
use std::{fs, path::PathBuf};

// ---------------------------------------------------------------------------
// Serde structs for Ollama manifests
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct OllamaManifest {
    pub(crate) layers: Vec<OllamaLayer>,
}

#[derive(Deserialize)]
pub(crate) struct OllamaLayer {
    #[serde(rename = "mediaType")]
    pub(crate) media_type: String,
    pub(crate) digest: String,
}

/// Media type of the GGUF model layer in an Ollama manifest.
pub(crate) const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// The shared on-disk model store (Ollama-compatible).
pub struct Store {
    pub root: PathBuf,
}

impl Store {
    /// Locate the store root: `$INFR_MODELS`, else `$XDG_CACHE_HOME/infr/models` (`~/.cache/infr/models`).
    /// Always our own writable dir — we never touch the system Ollama dirs. Need not exist yet.
    pub fn discover() -> Result<Self> {
        let root = if let Ok(p) = std::env::var("INFR_MODELS") {
            PathBuf::from(p)
        } else {
            dirs::cache_dir()
                .ok_or_else(|| Error::Other("cannot determine cache directory".into()))?
                .join("infr")
                .join("models")
        };
        Ok(Store { root })
    }

    /// Canonical `<namespace>/<name>` for an ollama ref: bare names get the `library/` namespace.
    pub(crate) fn ollama_full_name(name: &str) -> String {
        if name.contains('/') {
            name.to_string()
        } else {
            format!("library/{name}")
        }
    }

    /// Manifest path for an ollama ref: `<root>/manifests/registry.ollama.ai/<ns>/<name>/<tag>`.
    pub(crate) fn ollama_manifest_path(&self, name: &str, tag: &str) -> PathBuf {
        self.root
            .join("manifests")
            .join("registry.ollama.ai")
            .join(Self::ollama_full_name(name))
            .join(tag)
    }

    /// Manifest path for an hf ref: `<root>/manifests/huggingface.co/<repo>/<file>`.
    pub(crate) fn hf_manifest_path(&self, repo: &str, file: &str) -> PathBuf {
        self.root
            .join("manifests")
            .join("huggingface.co")
            .join(repo)
            .join(file)
    }

    /// Return the blobs directory (`<root>/blobs`).
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// If the referenced model already exists in our store, return the GGUF blob path.
    ///
    /// - `Path(p)` → `Some(p)` if the file exists.
    /// - `Hf`     → read the cached hf manifest (needs a known filename; `file: None` → `None`).
    /// - `Ollama` → read the cached ollama manifest, find the model layer, return the blob.
    pub fn resolve(&self, r: &ModelRef) -> Result<Option<PathBuf>> {
        match r {
            ModelRef::Path(p) => Ok(p.exists().then(|| p.clone())),
            ModelRef::Hf {
                repo,
                file: Some(f),
            } => self.blob_if_manifest(&self.hf_manifest_path(repo, f)),
            // No filename → we can't name the manifest without the HF API; `pull` resolves it.
            ModelRef::Hf { file: None, .. } => Ok(None),
            ModelRef::Ollama { name, tag } => {
                self.blob_if_manifest(&self.ollama_manifest_path(name, tag))
            }
        }
    }

    /// Read a manifest at `path` (if present) and return its model-layer blob (if present).
    fn blob_if_manifest(&self, path: &std::path::Path) -> Result<Option<PathBuf>> {
        if path.exists() {
            self.blob_from_manifest(path)
        } else {
            Ok(None)
        }
    }

    /// Parse a manifest file and return the GGUF blob path if it exists.
    fn blob_from_manifest(&self, manifest_path: &std::path::Path) -> Result<Option<PathBuf>> {
        let content = fs::read_to_string(manifest_path).map_err(|e| {
            Error::Other(format!("reading manifest {}: {e}", manifest_path.display()))
        })?;

        let manifest: OllamaManifest = serde_json::from_str(&content)
            .map_err(|e| Error::Other(format!("parsing manifest: {e}")))?;

        let model_layer = match manifest
            .layers
            .iter()
            .find(|l| l.media_type == "application/vnd.ollama.image.model")
        {
            Some(l) => l,
            None => return Ok(None),
        };

        // digest is "sha256:abc123…" → blob filename is "sha256-abc123…"
        let blob_name = model_layer.digest.replace(':', "-");
        let blob_path = self.root.join("blobs").join(blob_name);

        if blob_path.exists() {
            Ok(Some(blob_path))
        } else {
            Ok(None)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_manifest(digest: &str) -> String {
        serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.ollama.image.config",
                "digest": digest,
                "size": 0
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": digest,
                    "size": 42
                }
            ]
        })
        .to_string()
    }

    /// Write a manifest + blob in a temp store and assert resolve finds the blob.
    #[test]
    fn resolve_ollama_simple_name() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:aabbccddeeff001122334455667788990011223344556677889900aabbccddeeff";
        let blob_name = digest.replace(':', "-");

        // Manifest lives at: <root>/manifests/registry.ollama.ai/library/testmodel/latest
        // The tag ("latest") is the filename, not a directory.
        let manifest_parent = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("testmodel");
        fs::create_dir_all(&manifest_parent).unwrap();
        fs::write(manifest_parent.join("latest"), fake_manifest(digest)).unwrap();

        // Write blob
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        fs::write(blobs_dir.join(&blob_name), b"fake gguf data").unwrap();

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "testmodel".into(),
            tag: "latest".into(),
        };
        let result = store.resolve(&mr).unwrap();
        assert!(result.is_some(), "expected blob path, got None");
        assert!(result.unwrap().exists());
    }

    /// A namespaced Ollama ref should resolve via the secondary path.
    #[test]
    fn resolve_ollama_namespaced() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let blob_name = digest.replace(':', "-");

        // Write manifest at registry.ollama.ai/library/qwen2.5/latest
        // (secondary path for name="library/qwen2.5")
        let manifest_dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("qwen2.5");
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::write(manifest_dir.join("latest"), fake_manifest(digest)).unwrap();

        // Write blob
        let blobs_dir = root.join("blobs");
        fs::create_dir_all(&blobs_dir).unwrap();
        fs::write(blobs_dir.join(&blob_name), b"fake gguf").unwrap();

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "library/qwen2.5".into(),
            tag: "latest".into(),
        };
        let result = store.resolve(&mr).unwrap();
        assert!(result.is_some(), "expected blob path, got None");
    }

    /// Missing model (no manifest file) should return Ok(None).
    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Ollama {
            name: "doesnotexist".into(),
            tag: "latest".into(),
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// Manifest present but blob missing → Ok(None).
    #[test]
    fn resolve_manifest_no_blob() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();

        let digest = "sha256:0000000000000000000000000000000000000000000000000000000000000001";
        let manifest_dir = root
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("ghostmodel");
        fs::create_dir_all(&manifest_dir).unwrap();
        fs::write(manifest_dir.join("v1"), fake_manifest(digest)).unwrap();
        // intentionally do NOT create the blob

        let store = Store { root };
        let mr = ModelRef::Ollama {
            name: "ghostmodel".into(),
            tag: "v1".into(),
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// HF refs always resolve to None (network-only).
    #[test]
    fn resolve_hf_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Hf {
            repo: "org/repo".into(),
            file: None,
        };
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }

    /// Path variant returns Some when file exists.
    #[test]
    fn resolve_path_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("model.gguf");
        fs::write(&gguf, b"fake").unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Path(gguf.clone());
        assert_eq!(store.resolve(&mr).unwrap(), Some(gguf));
    }

    /// Path variant returns None when file is absent.
    #[test]
    fn resolve_path_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store {
            root: tmp.path().to_path_buf(),
        };
        let mr = ModelRef::Path(tmp.path().join("nofile.gguf"));
        assert_eq!(store.resolve(&mr).unwrap(), None);
    }
}
