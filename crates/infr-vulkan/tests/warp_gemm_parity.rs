//! Bitwise parity for the Q4K warp GEMM A_GLOBAL pairs (n128_ag and sk_ag, resident vs
//! `-DSTREAMED` twin) — the prefill-hot tile variants gemv_streamed_parity.rs doesn't cover.
//! Born as the resident-BDA perf campaign's ISA probe and kept for the coverage; it still doubles
//! as the ISA-dump vehicle: RADV_DEBUG=shaders MESA_SHADER_CACHE_DISABLE=true <bin> --ignored
//! 2> isa.txt (move ~/.cache/infr/vk-pipeline-cache-*.bin aside first).
use infr_core::backend::{Backend, BufferUsage};
use infr_core::DType;
use infr_vulkan::VulkanBackend;

fn synth_bytes(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) >> 7;
            (h % 0x40) as u8
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn warp_ag_isa_probe() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    let dtype = DType::Q4K;
    let (m, k, n, splits) = (64usize, 512usize, 256usize, 2usize);
    let mpad = 64usize;
    let w_bytes = n * k / 256 * 144; // Q4K: 256 elems / 144 bytes

    let w = synth_bytes(w_bytes, 7);
    let a16 = synth_bytes(mpad * k * 2, 13); // f16 bits, high byte < 0x40 => finite

    let a_buf = be.alloc(a16.len(), BufferUsage::Activations).unwrap();
    be.upload(a_buf.as_ref(), &a16).unwrap();
    let c_buf = be.alloc(m * n * 4, BufferUsage::Activations).unwrap();
    let part_buf = be
        .alloc(splits * mpad * n * 4, BufferUsage::Activations)
        .unwrap();

    let w_buf = be.alloc(w_bytes, BufferUsage::Weights).unwrap();
    be.upload(w_buf.as_ref(), &w).unwrap();
    let (arena, addr) = be.alloc_arena_bda(w_bytes).unwrap();
    be.upload(arena.as_ref(), &w).unwrap();

    let mut outs: Vec<Vec<u8>> = Vec::new();
    // n128_ag resident, n128_ag streamed, sk_ag resident, sk_ag streamed
    for (sk, ar) in [
        (false, None),
        (false, Some(addr)),
        (true, None),
        (true, Some(addr)),
    ] {
        let rec = be.recorder().unwrap();
        if sk {
            rec.matmul_native_splitk(
                dtype,
                a_buf.as_ref(),
                w_buf.as_ref(),
                0,
                part_buf.as_ref(),
                c_buf.as_ref(),
                m,
                k,
                n,
                splits,
                true,
                ar,
            );
        } else {
            rec.matmul_native_f16a(
                dtype,
                a_buf.as_ref(),
                w_buf.as_ref(),
                0,
                c_buf.as_ref(),
                m,
                k,
                n,
                ar,
            );
        }
        rec.finish().unwrap();
        let mut out = vec![0u8; m * n * 4];
        be.download(c_buf.as_ref(), &mut out).unwrap();
        assert!(
            out.iter().any(|&b| b != 0),
            "all-zero output (sk={sk} ar={ar:?})"
        );
        outs.push(out);
    }
    assert_eq!(outs[0], outs[1], "n128_ag streamed != resident (bitwise)");
    assert_eq!(outs[2], outs[3], "sk_ag streamed != resident (bitwise)");
    println!("ok: q4k n128_ag + sk_ag streamed pairs bitwise-match resident");
}
