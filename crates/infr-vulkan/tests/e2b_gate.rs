//! Integration test for e2b_gate kernel.
use infr_vulkan::VulkanBackend;

fn gelu_cpu(x: f32) -> f32 {
    0.5 * x
        * (1.0
            + (0.7978845608028654f64 * (x as f64 + 0.044715f64 * x as f64 * x as f64 * x as f64))
                .tanh()) as f32
}

#[test]
#[ignore = "requires a Vulkan GPU"]
fn e2b_gate_full() {
    let be = VulkanBackend::new().unwrap();
    let m = 1usize;
    let in_f = 1536usize;
    let out_f = 256usize;
    let up_off = 0usize;
    let up_stride = out_f; // non-strided for test

    // hidden = all 1.0
    let hidden: Vec<f32> = vec![1.0f32; m * in_f];
    // weight[o,k] = (o+1) → GEMV result y[o] = in_f * (o+1)
    let mut weight = vec![0f32; out_f * in_f];
    for o in 0..out_f {
        for k in 0..in_f {
            weight[o * in_f + k] = (o + 1) as f32;
        }
    }
    // ipl = 1.0 for all
    let ipl: Vec<f32> = vec![1.0f32; m * out_f];

    // CPU reference
    let expected: Vec<f32> = (0..out_f)
        .map(|o| {
            let gemv = in_f as f32 * (o + 1) as f32;
            let up_idx = up_off + o;
            gelu_cpu(gemv) * ipl[up_idx]
        })
        .collect();

    let got = be
        .e2b_gate(&weight, &hidden, &ipl, up_off, up_stride, m, in_f, out_f)
        .unwrap();

    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        let diff = (e - g).abs();
        if diff > 1e-6 {
            panic!("mismatch at {i}: expected {e:.6} got {g:.6} diff {diff:.2e}");
        }
    }
    eprintln!(
        "OK: e2b_gate matches CPU reference ({} elements)",
        got.len()
    );
}
