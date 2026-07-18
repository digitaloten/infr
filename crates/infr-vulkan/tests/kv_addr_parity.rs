//! Proves the scalar decode attention kernel (`attention_kv`) reads its K/V cache correctly by
//! 64-bit device address (`kv_addr.glsl`) — the KV-side mirror of `weight_addr_parity.rs`. The
//! `-DKV_BDA` twin (`attention_kv_at`) must compute BIT-IDENTICALLY to the bound-SSBO build
//! (`attention_kv`): the two differ ONLY in where K/V words come from (a bound binding vs a
//! `k_addr`/`v_addr` pointer read), the softmax/accumulation math is the same source, so anything
//! short of bitwise equality is a mis-addressing bug, not a tolerance question.
//!
//! Two assertions per case:
//!  * BOUND-VS-POINTER: `attention_kv` (K/V bound at slots 1/2) vs `attention_kv_at` (K/V at their
//!    own device address, arena offset 0) — identical bits.
//!  * OFFSET-INVARIANCE (the load-bearing one): the SAME K/V bytes parked behind a garbage non-zero
//!    prefix inside a `KvCache` buffer, with `k_addr`/`v_addr` = base+prefix. A twin that ignored
//!    the offset (or mishandled the 64-bit add's non-zero low bits) would read the garbage prefix
//!    and diverge — this is what integration's per-tensor arena offsets need to work.
//!
//! Covers f16 K/V and all three planar-Q8 combos (K-only, V-only, both). Q/K/V bytes are drawn so
//! every f16 decodes finite and non-degenerate (high byte < 0x40 → sign+top-exp bit clear, never
//! NaN/Inf — a NaN output would make a bitwise compare pass vacuously, hiding the mis-addressing
//! this test exists to catch); Q8 codes/scales use the same < 0x40 range.
//!
//! Run: `cargo test -p infr-vulkan --test kv_addr_parity -- --ignored --nocapture`
//! ISA probe: `RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true cargo test -p infr-vulkan \
//!   --test kv_addr_parity kv_isa_probe -- --ignored --nocapture 2> isa.txt` (move the pipeline
//!   cache aside first).
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

/// Pseudo-random bytes in `0x00..=0x3F` — Q8 codes (small +ints) and scales (finite +f16).
fn synth_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

/// `n` f16 elements whose bits are masked to `0x3FFF`: sign + top exponent bit clear → every value
/// is finite, non-negative, < 2.0. Never NaN/Inf, so a bitwise output compare is never vacuous.
fn synth_f16(n: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let h = ((i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 5) as u16;
        out.extend_from_slice(&(h & 0x3FFF).to_le_bytes());
    }
    out
}

fn bits(v: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(v).to_vec()
}

#[derive(Clone, Copy)]
struct Shape {
    q_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
}

/// Number of f16 elements in one KV-cache side for a full-context (rcap=0) run at pos_offset 0:
/// query row `q_len-1` attends key `q_len-1`, so the cache spans `q_len` rows of `nkv*hd`.
fn kv_elems(s: Shape) -> usize {
    s.q_len * s.nkv * s.hd
}

/// Bytes of one KV side: f16 = 2/elem; planar Q8_0 = codes[cap] (1 byte) + scales[cap/32] (f16).
fn side_bytes(s: Shape, q8: bool) -> usize {
    let cap = kv_elems(s);
    if q8 {
        assert_eq!(
            cap % 32,
            0,
            "q8 cap must be a whole number of 32-elem blocks"
        );
        cap + (cap / 32) * 2
    } else {
        cap * 2
    }
}

/// Synth bytes for one KV side (differ per side via `seed` so a K/V swap would show up).
fn side_data(s: Shape, q8: bool, seed: usize) -> Vec<u8> {
    if q8 {
        synth_bytes(side_bytes(s, q8), seed)
    } else {
        synth_f16(kv_elems(s), seed)
    }
}

/// 256-byte garbage prefix (aligned for f16 & u32-word reads) filled with DIFFERENT bytes, so a
/// kernel that reads from the buffer base instead of base+off gets visibly wrong data.
const OFF: usize = 256;

struct Legs {
    bound: Vec<u32>,
    at0: Vec<u32>,
    atoff: Vec<u32>,
}

/// Runs all three legs (bound, pointer@0, pointer@nonzero-offset) for one shape/dtype combo.
fn run(be: &VulkanBackend, s: Shape, k_q8: bool, v_q8: bool) -> Legs {
    let (q_len, nh, nkv, hd) = (s.q_len, s.nh, s.nkv, s.hd);
    let cap = if k_q8 || v_q8 { kv_elems(s) } else { 0 };
    let scale = 0.0f32; // default 1/sqrt(hd)
    let window = 0usize; // full causal
    let pos = 0usize;

    let q = synth_f16(q_len * nh * hd, 101);
    let q_buf = be.alloc(q.len(), BufferUsage::Activations).unwrap();
    be.upload(q_buf.as_ref(), &q).unwrap();
    let o_bytes = q_len * nh * hd * 4;

    let kd = side_data(s, k_q8, 11);
    let vd = side_data(s, v_q8, 23);

    // ── Leg A: bound SSBOs (K/V at slots 1/2) ────────────────────────────────────────────────
    let kbuf = be.alloc(kd.len(), BufferUsage::KvCache).unwrap();
    let vbuf = be.alloc(vd.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf.as_ref(), &kd).unwrap();
    be.upload(vbuf.as_ref(), &vd).unwrap();
    let o_buf = be.alloc(o_bytes, BufferUsage::Activations).unwrap();
    let rec = be.recorder().unwrap();
    rec.attention_kv(
        q_buf.as_ref(),
        kbuf.as_ref(),
        vbuf.as_ref(),
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let bound = bits(&out);

    // ── Leg B: pointer read, K/V at arena offset 0 ───────────────────────────────────────────
    let ka0 = kbuf
        .device_addr()
        .expect("KvCache K must expose a device address");
    let va0 = vbuf
        .device_addr()
        .expect("KvCache V must expose a device address");
    let rec = be.recorder().unwrap();
    rec.attention_kv_at(
        q_buf.as_ref(),
        kbuf.as_ref(),
        vbuf.as_ref(),
        ka0,
        va0,
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let at0 = bits(&out);

    // ── Leg C: pointer read, SAME bytes parked behind a garbage prefix in a KvCache buffer ────
    let mut kback = synth_bytes(OFF, 0xBAD);
    kback.extend_from_slice(&kd);
    let mut vback = synth_bytes(OFF, 0xBEEF);
    vback.extend_from_slice(&vd);
    let kbuf2 = be.alloc(kback.len(), BufferUsage::KvCache).unwrap();
    let vbuf2 = be.alloc(vback.len(), BufferUsage::KvCache).unwrap();
    be.upload(kbuf2.as_ref(), &kback).unwrap();
    be.upload(vbuf2.as_ref(), &vback).unwrap();
    let ka = kbuf2.device_addr().unwrap() + OFF as u64;
    let va = vbuf2.device_addr().unwrap() + OFF as u64;
    let rec = be.recorder().unwrap();
    rec.attention_kv_at(
        q_buf.as_ref(),
        kbuf2.as_ref(),
        vbuf2.as_ref(),
        ka,
        va,
        o_buf.as_ref(),
        q_len,
        q_len,
        nh,
        nkv,
        hd,
        pos,
        window,
        scale,
        k_q8,
        v_q8,
        cap,
    );
    rec.finish().unwrap();
    let mut out = vec![0u8; o_bytes];
    be.download(o_buf.as_ref(), &mut out).unwrap();
    let atoff = bits(&out);

    Legs { bound, at0, atoff }
}

fn assert_legs(name: &str, l: &Legs) {
    assert!(
        l.bound.iter().any(|&b| b != 0),
        "{name}: bound output is all zeros — the case is not exercising the kernel"
    );
    for (i, (&b, &p)) in l.bound.iter().zip(l.at0.iter()).enumerate() {
        assert_eq!(
            b, p,
            "{name}: pointer@0 differs from bound at out {i}: {} vs {} (bits {b:#010x} vs {p:#010x})",
            f32::from_bits(b),
            f32::from_bits(p)
        );
    }
    for (i, (&b, &p)) in l.bound.iter().zip(l.atoff.iter()).enumerate() {
        assert_eq!(
            b,
            p,
            "{name}: pointer@nonzero-offset differs from bound at out {i}: {} vs {} — the twin is \
             ignoring its K/V base offset, which breaks every KV tensor but one at a shared base",
            f32::from_bits(b),
            f32::from_bits(p)
        );
    }
}

const SHAPES: [Shape; 4] = [
    Shape {
        q_len: 4,
        nh: 8,
        nkv: 2,
        hd: 64,
    },
    Shape {
        q_len: 1,
        nh: 4,
        nkv: 4,
        hd: 128,
    },
    Shape {
        q_len: 3,
        nh: 8,
        nkv: 8,
        hd: 64,
    },
    Shape {
        q_len: 6,
        nh: 8,
        nkv: 1,
        hd: 64,
    },
];

#[test]
#[ignore = "requires a Vulkan GPU"]
fn attention_kv_bda_matches_bound_f16() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    for s in SHAPES {
        let l = run(&be, s, false, false);
        let name = format!(
            "f16 q_len={} nh={} nkv={} hd={}",
            s.q_len, s.nh, s.nkv, s.hd
        );
        assert_legs(&name, &l);
        println!("ok: {name}");
    }
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn attention_kv_bda_matches_bound_q8() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // K-only, V-only, and both — each Q8 side reads its own k_addr/v_addr through kv_word().
    for (k_q8, v_q8) in [(true, false), (false, true), (true, true)] {
        for s in SHAPES {
            let l = run(&be, s, k_q8, v_q8);
            let name = format!(
                "q8(k={k_q8},v={v_q8}) q_len={} nh={} nkv={} hd={}",
                s.q_len, s.nh, s.nkv, s.hd
            );
            assert_legs(&name, &l);
            println!("ok: {name}");
        }
    }
}

/// ISA-dump vehicle (mirrors `warp_gemm_parity::warp_ag_isa_probe`): dispatches ONLY the `-DKV_BDA`
/// pointer build so `RADV_DEBUG=shaders` emits its ISA. Confirm the inner K/V loads use a
/// scalar/saddr base + 32-bit offset (`global_load_*` with an s[]/saddr base), NOT a per-load
/// `v_add_co_u32`/`v_addc` 64-bit address materialization — the shape that measured the 2.4x
/// regression on the weight side.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn kv_isa_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let s = Shape {
        q_len: 8,
        nh: 8,
        nkv: 2,
        hd: 128,
    };
    let l = run(&be, s, false, false);
    assert_legs("isa-probe f16", &l);
    println!("ok: kv_isa_probe (dispatched attention_kv_bda)");
}
