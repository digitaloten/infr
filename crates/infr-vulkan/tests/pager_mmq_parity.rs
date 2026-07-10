//! Proves `matmul_mmq_experts_paged` (the batched-MoE dp4a expert GEMM reading a `GpuPager`
//! arena through the word-base LUT — Scout's batched paged prefill) matches a host reference,
//! INCLUDING under eviction churn: experts are made resident, evicted by touching others, and
//! re-uploaded into DIFFERENT slots before the routed dispatch — so a slot-index/word-base mixup
//! or a stale-LUT read produces wrong numbers, not a crash (the coherent-but-wrong bug class the
//! `NW(i)` doc in native_decode.glsl records). Covers both new paged dtypes in one pipeline
//! (Q2_K gate GEMM, Q3_K down-style GEMM on the gate's output re-quantized) at a ragged bucket
//! layout (counts 4/3/2 over 3 of 5 experts, two experts absent — their LUT entries stale).
//!
//! Run: `cargo test -p infr-vulkan --test pager_mmq_parity -- --ignored --nocapture`
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::pager::GpuPager;
use infr_vulkan::VulkanBackend;

// ---- Q2_K / Q3_K synthetic encoders + reference decoders. Same internal-round-trip-only
// contract as the adapter's test helpers (self-consistent test-data encoders whose DECODERS
// match the GPU shaders' layout exactly), duplicated here because those helpers live in
// adapter.rs's #[cfg(test)] mod.

fn q2_k(x: &[f32]) -> Vec<u8> {
    let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
    let shift = |si: usize| 2 * ((si & 7) >> 1);
    let mut out = Vec::with_capacity(x.len() / 256 * 84);
    for blk in x.chunks(256) {
        let mut sub_lo = [0f32; 16];
        let mut sub_sc = [0f32; 16];
        for (si, sub) in blk.chunks(16).enumerate() {
            let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            sub_lo[si] = lo;
            sub_sc[si] = ((hi - lo) / 3.0).max(1e-8);
        }
        let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 15.0;
        let dmin = sub_lo
            .iter()
            .cloned()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(1e-8)
            / 15.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let idmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
        let mut scales = [0u8; 16];
        let mut sc = [0u32; 16];
        let mut mn = [0u32; 16];
        for si in 0..16 {
            sc[si] = ((sub_sc[si] * id).round() as i32).clamp(0, 15) as u32;
            mn[si] = ((sub_lo[si].abs() * idmin).round() as i32).clamp(0, 15) as u32;
            scales[si] = (sc[si] | (mn[si] << 4)) as u8;
        }
        out.extend_from_slice(&scales);
        let mut qs = [0u8; 64];
        for si in 0..16 {
            let scale = d * sc[si] as f32;
            let min = dmin * mn[si] as f32;
            let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for l in 0..16 {
                let v = blk[si * 16 + l];
                let q = (((v - min) * iscale).round() as i32).clamp(0, 3) as u8;
                qs[base(si) + l] |= q << shift(si);
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
    }
    out
}
fn deq_q2_k(bytes: &[u8]) -> Vec<f32> {
    let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
    let shift = |si: usize| 2 * ((si & 7) >> 1);
    let mut out = Vec::with_capacity(bytes.len() / 84 * 256);
    for blk in bytes.chunks(84) {
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = half::f16::from_le_bytes([blk[80], blk[81]]).to_f32();
        let dmin = half::f16::from_le_bytes([blk[82], blk[83]]).to_f32();
        for si in 0..16usize {
            let scale = d * (scales[si] & 0xF) as f32;
            let min = dmin * (scales[si] >> 4) as f32;
            for l in 0..16 {
                let q = (qs[base(si) + l] >> shift(si)) & 3;
                out.push(scale * q as f32 - min);
            }
        }
    }
    out
}

fn q3_k(x: &[f32]) -> Vec<u8> {
    let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
    let shift = |si: usize| 2 * ((si & 7) >> 1);
    let hbit = |si: usize| 4 * (si >> 3) + ((si & 7) >> 1);
    let mut out = Vec::with_capacity(x.len() / 256 * 110);
    for blk in x.chunks(256) {
        let mut sub_sc = [0f32; 16];
        for (si, sub) in blk.chunks(16).enumerate() {
            let amax = sub.iter().cloned().fold(0f32, |m, v| m.max(v.abs()));
            sub_sc[si] = (amax / 3.0).max(1e-8);
        }
        let d = sub_sc.iter().cloned().fold(0f32, f32::max) / 31.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let mut s6 = [0i32; 16];
        for si in 0..16 {
            s6[si] = ((sub_sc[si] * id).round() as i32).clamp(1, 31);
        }
        let mut sr = [0u8; 12];
        for (si, &s) in s6.iter().enumerate() {
            let val = (s + 32) as u8;
            let k = si >> 2;
            let bi = si & 3;
            let lo4 = val & 0xF;
            let hi2 = (val >> 4) & 3;
            match k {
                0 => {
                    sr[bi] |= lo4;
                    sr[8 + bi] |= hi2;
                }
                1 => {
                    sr[4 + bi] |= lo4;
                    sr[8 + bi] |= hi2 << 2;
                }
                2 => {
                    sr[bi] |= lo4 << 4;
                    sr[8 + bi] |= hi2 << 4;
                }
                _ => {
                    sr[4 + bi] |= lo4 << 4;
                    sr[8 + bi] |= hi2 << 6;
                }
            }
        }
        let hmask_start = out.len();
        out.extend_from_slice(&[0u8; 32]);
        let mut qs = [0u8; 64];
        for si in 0..16 {
            let scale = d * s6[si] as f32;
            let iscale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for l in 0..16 {
                let v = blk[si * 16 + l];
                let q3u = (((v * iscale) + 4.0).round() as i32).clamp(0, 7) as u8;
                qs[base(si) + l] |= (q3u & 3) << shift(si);
                if q3u & 4 != 0 {
                    out[hmask_start + 16 * (si & 1) + l] |= 1 << hbit(si);
                }
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&sr);
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    out
}
fn deq_q3_k(bytes: &[u8]) -> Vec<f32> {
    let base = |si: usize| 32 * (si >> 3) + 16 * (si & 1);
    let shift = |si: usize| 2 * ((si & 7) >> 1);
    let hbit = |si: usize| 4 * (si >> 3) + ((si & 7) >> 1);
    let sc3 = |sr: &[u8], si: usize| -> i32 {
        let k = si >> 2;
        let bi = si & 3;
        let (a, b, c) = (sr[bi] as u32, sr[4 + bi] as u32, sr[8 + bi] as u32);
        let val = match k {
            0 => (a & 0xF) | ((c & 3) << 4),
            1 => (b & 0xF) | (((c >> 2) & 3) << 4),
            2 => ((a >> 4) & 0xF) | (((c >> 4) & 3) << 4),
            _ => ((b >> 4) & 0xF) | (((c >> 6) & 3) << 4),
        };
        val as i32 - 32
    };
    let mut out = Vec::with_capacity(bytes.len() / 110 * 256);
    for blk in bytes.chunks(110) {
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let sr = &blk[96..108];
        let d = half::f16::from_le_bytes([blk[108], blk[109]]).to_f32();
        for si in 0..16usize {
            let scale = d * sc3(sr, si) as f32;
            for l in 0..16 {
                let low2 = (qs[base(si) + l] >> shift(si)) & 3;
                let high = (hmask[16 * (si & 1) + l] >> hbit(si)) & 1;
                out.push(scale * ((low2 | (high << 2)) as f32 - 4.0));
            }
        }
    }
    out
}

/// Host mirror of `quant_q8` (per-32-block symmetric int8): the GEMM's activation operand.
fn quant_act(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = half::f16::from_f32(amax / 127.0).to_f32();
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in blk {
            out.push((v * id).round().clamp(-127.0, 127.0) * d);
        }
    }
    out
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn paged_mmq_expert_gemm_matches_host_under_eviction() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // NB: no i8_dot capability gate here — the production caller (`execute_paged_moe`'s batched
    // arm) checks `caps().i8_dot` before choosing this kernel; a non-dp4a device would skip.
    // k=256 (one Q2_K/Q3_K superblock), n=64 (one BN tile) — the smallest legal GEMM. 5 experts,
    // only 3 arena slots per pager: the churn sequence below forces evictions and slot reuse.
    let (k, n, n_expert) = (256usize, 64usize, 5usize);
    let stride_elems = k * n;
    let gate_slot_bytes = stride_elems / 256 * 84; // Q2_K
    let down_slot_bytes = stride_elems / 256 * 110; // Q3_K
    let f = |i: usize, s: f32| (i as f32 * s).sin() * 0.15;

    let gate_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| {
            let w: Vec<f32> = (0..stride_elems).map(|i| f(i + e * 977, 0.017)).collect();
            q2_k(&w)
        })
        .collect();
    let down_banks: Vec<Vec<u8>> = (0..n_expert)
        .map(|e| {
            let w: Vec<f32> = (0..stride_elems).map(|i| f(i + e * 977, 0.029)).collect();
            q3_k(&w)
        })
        .collect();
    let gate_host: Vec<Vec<f32>> = gate_banks.iter().map(|b| deq_q2_k(b)).collect();
    let down_host: Vec<Vec<f32>> = down_banks.iter().map(|b| deq_q3_k(b)).collect();

    let mut gate_pager = GpuPager::new(&be, n_expert, 3, gate_slot_bytes).unwrap();
    let mut down_pager = GpuPager::new(&be, n_expert, 3, down_slot_bytes).unwrap();
    let staging = be
        .alloc_uninit(gate_slot_bytes.max(down_slot_bytes), BufferUsage::Staging)
        .unwrap();

    // Churn: fill all 3 slots with experts 0/1/2, then route to {2,3,4} — 3 and 4 evict 0 and 1
    // (LRU), landing in REUSED slots, and 2 stays where it was. The final LUT therefore maps the
    // routed experts to a scrambled slot order; experts 0/1's LUT entries go stale (never read —
    // their bucket counts are 0).
    for pre in [0u32, 1, 2] {
        gate_pager
            .ensure_resident(&be, staging.as_ref(), pre, &gate_banks[pre as usize])
            .unwrap();
        down_pager
            .ensure_resident(&be, staging.as_ref(), pre, &down_banks[pre as usize])
            .unwrap();
    }
    let routed = [2u32, 3, 4];
    for &eid in &routed {
        gate_pager
            .ensure_resident(&be, staging.as_ref(), eid, &gate_banks[eid as usize])
            .unwrap();
        down_pager
            .ensure_resident(&be, staging.as_ref(), eid, &down_banks[eid as usize])
            .unwrap();
    }
    gate_pager.flush_lut(&be).unwrap();
    down_pager.flush_lut(&be).unwrap();

    // Ragged bucket layout over the routed experts: counts[2]=4, counts[3]=3, counts[4]=2.
    let counts_host: Vec<u32> = vec![0, 0, 4, 3, 2];
    let offsets_host: Vec<u32> = vec![0, 0, 0, 4, 7];
    let n_pairs = 9usize;
    let npad = n_pairs.div_ceil(64) * 64 + 64; // GEMM As-stage overread padding

    let x: Vec<f32> = (0..n_pairs * k).map(|i| f(i, 0.11) + 0.02).collect();
    let xq = quant_act(&x);

    let mk_u32 = |v: &[u32]| {
        let b = be.alloc(v.len() * 4, BufferUsage::Activations).unwrap();
        be.upload(b.as_ref(), bytemuck::cast_slice(v)).unwrap();
        b
    };
    let counts = mk_u32(&counts_host);
    let offsets = mk_u32(&offsets_host);
    let xbuf = be.alloc(n_pairs * k * 4, BufferUsage::Activations).unwrap();
    be.upload(xbuf.as_ref(), bytemuck::cast_slice(&x)).unwrap();
    // zero-init alloc: the padded overread rows read zeros, results discarded at the store.
    let qa = be.alloc(npad * k, BufferUsage::Activations).unwrap();
    let qda = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let qsa = be
        .alloc(npad * (k / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let gbuf = be.alloc(npad * n * 4, BufferUsage::Activations).unwrap();
    let dqa = be.alloc(npad * n, BufferUsage::Activations).unwrap();
    let dda = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let dsa = be
        .alloc(npad * (n / 32) * 2, BufferUsage::Activations)
        .unwrap();
    let ybuf = be.alloc(npad * k * 4, BufferUsage::Activations).unwrap();

    // ONE recorder, chained: quant → Q2_K paged GEMM → re-quant → Q3_K paged GEMM (a down-style
    // GEMM whose k is the gate's n) — the dependency chain catches missing barriers, not just
    // wrong math (the pager_gemv_multi_parity lesson).
    let rec = be.recorder().unwrap();
    rec.quant_q8(
        xbuf.as_ref(),
        qa.as_ref(),
        qda.as_ref(),
        qsa.as_ref(),
        n_pairs,
        k,
    );
    rec.matmul_mmq_experts_paged(
        DType::Q2K,
        "expert_gateup",
        qa.as_ref(),
        qda.as_ref(),
        gate_pager.arena_buffer(),
        gate_pager.lut_buffer(),
        0, // single "layer": layer_base 0, local id == global id
        counts.as_ref(),
        offsets.as_ref(),
        gbuf.as_ref(),
        n_pairs,
        k,
        n,
        n_expert,
        1,
    );
    rec.quant_q8(
        gbuf.as_ref(),
        dqa.as_ref(),
        dda.as_ref(),
        dsa.as_ref(),
        n_pairs,
        n,
    );
    rec.matmul_mmq_experts_paged(
        DType::Q3K,
        "expert_down",
        dqa.as_ref(),
        dda.as_ref(),
        down_pager.arena_buffer(),
        down_pager.lut_buffer(),
        0,
        counts.as_ref(),
        offsets.as_ref(),
        ybuf.as_ref(),
        n_pairs,
        n,
        k,
        n_expert,
        1,
    );
    rec.finish().unwrap();

    let mut g_out = vec![0f32; npad * n];
    be.download(gbuf.as_ref(), bytemuck::cast_slice_mut(&mut g_out))
        .unwrap();
    let mut y_out = vec![0f32; npad * k];
    be.download(ybuf.as_ref(), bytemuck::cast_slice_mut(&mut y_out))
        .unwrap();

    // Host reference, mirroring both quantization layers.
    let mut want_g = vec![0f32; n_pairs * n];
    for e in 0..n_expert {
        let (off, cnt) = (offsets_host[e] as usize, counts_host[e] as usize);
        for r in off..off + cnt {
            let xr = &xq[r * k..(r + 1) * k];
            for o in 0..n {
                want_g[r * n + o] = gate_host[e][o * k..(o + 1) * k]
                    .iter()
                    .zip(xr)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
    }
    for i in 0..n_pairs * n {
        assert!(
            (g_out[i] - want_g[i]).abs() < 5e-2,
            "paged Q2_K GEMM mismatch at {i}: got {} want {}",
            g_out[i],
            want_g[i]
        );
    }
    let gq = quant_act(&want_g);
    for e in 0..n_expert {
        let (off, cnt) = (offsets_host[e] as usize, counts_host[e] as usize);
        for r in off..off + cnt {
            let gr = &gq[r * n..(r + 1) * n];
            for o in 0..k {
                let want: f32 = down_host[e][o * n..(o + 1) * n]
                    .iter()
                    .zip(gr)
                    .map(|(a, b)| a * b)
                    .sum();
                let got = y_out[r * k + o];
                assert!(
                    (got - want).abs() < 5e-2,
                    "paged Q3_K GEMM mismatch row {r} out {o}: got {got} want {want}"
                );
            }
        }
    }
    println!("paged mmq expert GEMM under eviction OK");
}
