fn main() {
    // TODO(sonnet): compile shaders/*.comp -> SPIR-V here (e.g. via the `shaderc` crate),
    // or copy/reuse precompiled .spv from the ggml-vulkan build. Emit them to OUT_DIR and
    // include_bytes! them in src/. See PLAN.md "compute / Vulkan".
    println!("cargo:rerun-if-changed=shaders");
}
