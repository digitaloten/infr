// Buffer-device-address (BDA) access to the KV cache by its 64-bit device address — the KV-side
// twin of native_weight_addr.glsl (whose `w_addr`/`arena_word`/NW4 shapes this mirrors exactly).
// Included ONLY by the `-DKV_BDA` build of attention_kv.comp; the default (bound-SSBO) build never
// touches this header. See that shader's KV_BDA arm and tests/kv_addr_parity.rs.
//
// WHY TWO POINTERS: attention reads K and V independently (the QKᵀ dot loop, then the P·V weighted
// sum), off two distinct cache tensors at two distinct device addresses — so unlike the weight side
// (one `w_addr`) there are TWO wave-uniform base pointers, `k_addr` and `v_addr`. Each is passed
// from the host as a uvec2 push-constant pair (k_lo/k_hi, v_lo/v_hi) — a uvec2 split avoids the
// 8-byte push-constant alignment a uint64_t member would force, exactly like the weight side's
// arena_lo/arena_hi. main() sets both ONCE via `kv_base(lo, hi)`.
//
// ADDRESSING INVARIANT (same as the weight side): the u64 protects the cache tensor's BASE address;
// the intra-tensor element→byte offset is built entirely in u32 (<2^32 elements, <4 GiB per KV
// tensor at every supported context — the no-overflow claim on issue #74) and added to the base
// BEFORE the uint64_t cast, with a CONSTANT deref index. NIR then sees
// `iadd(uniform64, u2u64(divergent32))`, which ACO selects as a scalar-base global_load (64-bit
// base in SGPRs, one 32-bit VGPR offset) — NOT a per-load 64-bit VGPR address (the v_add_co /
// v_add_co_ci carry-add pair that measured the ~2.2-2.4x regression on the weight-side Q6K warp
// GEMM). The `base` args below are always fed a wave-uniform pointer (k_addr/v_addr, from push
// constants), so no explicit uniformity hint is needed — the iadd shape alone gets saddr selected.
//
// READ-ONLY: every buffer_reference here is `readonly`. The KV STORE kernels (store_f16/store_q8)
// are a LATER slice; a writable KV buffer_reference is deliberately NOT provided yet.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

uint64_t k_addr = 0ul; // K cache base byte address (set once in main from k_lo/k_hi)
uint64_t v_addr = 0ul; // V cache base byte address (set once in main from v_lo/v_hi)

uint64_t kv_base(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }

// ── Scalar reads (used by attention_kv this slice) ─────────────────────────────────────────────
// One f16 element `i` off `base` (the f16 KV cache read: `float(k[i])` / `float(v[i])`). Byte
// offset `i<<1` is built in u32 and added before the cast; deref index is the constant 0 → saddr.
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer KvHalf { float16_t v[]; };
float kv_half(uint64_t base, uint i) {
    return float(KvHalf(base + uint64_t(i << 1u)).v[0]);
}

// One u32 word `wi` off `base` — the planar-Q8 code/scale read (`ku[wi]` / `vu[wi]`). Identical
// shape to native_weight_addr.glsl's `arena_word`: byte offset `wi<<2` in u32, deref index 0.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvWords { uint v[]; };
uint kv_word(uint64_t base, uint wi) {
    return KvWords(base + uint64_t(wi << 2u)).v[0];
}

// ── Wide constant-index reads (declared now; consumed by slices 2+ which read f16vec4 KV) ───────
// The b128/b64 analogs of native_weight_addr.glsl's NW4/NW2: four (KV4) or two (KV2) CONSTANT
// indices off ONE pointer, so ACO's load/store vectorizer fuses them into a single
// global_load_b128 / global_load_b64 with a saddr scalar base — where N separate kv_word() calls
// would stay N unfused global_load_b32. `wbase` is a u32 WORD base (dword-aligned); the byte offset
// `wbase<<2` is built in u32 exactly as the scalar reads above. Requires only dword alignment on
// RDNA (b128/b64 both), matching buffer_reference_align.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvW4 { uint v[4]; };
uvec4 kv_word4(uint64_t base, uint wbase) {
    KvW4 p = KvW4(base + uint64_t(wbase << 2u));
    return uvec4(p.v[0], p.v[1], p.v[2], p.v[3]);
}
#define KV4(base, wbase) kv_word4(base, wbase)

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer KvW2 { uint v[2]; };
uvec2 kv_word2(uint64_t base, uint wbase) {
    KvW2 p = KvW2(base + uint64_t(wbase << 2u));
    return uvec2(p.v[0], p.v[1]);
}
#define KV2(base, wbase) kv_word2(base, wbase)
