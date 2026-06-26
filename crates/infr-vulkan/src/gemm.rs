//! Cooperative-matrix GEMM (the production matmul primitive). Uses the GLSL coopmat shader
//! compiled by build.rs. f16 inputs, f32 accumulate/output. v1 requires m,n,k multiples of 16.

use std::sync::OnceLock;

use ash::vk;
use half::f16;

use infr_core::{backend::BufferUsage, error::Result, Backend};

use super::{as_vk_buf, be, VulkanBackend};

const GEMM_SPV_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat.spv"));
static GEMM_SPV: OnceLock<Vec<u32>> = OnceLock::new();

fn gemm_spv() -> &'static [u32] {
    GEMM_SPV.get_or_init(|| {
        GEMM_SPV_BYTES
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    })
}

impl VulkanBackend {
    /// Cooperative-matrix GEMM: `C[m,n] = A[m,k] * B[k,n]` (row-major). `a`,`b` are f32 host
    /// slices (converted to f16 for the matrix cores); the result is f32. `m,n,k` must be
    /// multiples of 16 (v1).
    pub fn matmul_f16(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        assert!(
            m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
            "coopmat GEMM v1 needs m,n,k multiples of 16 (got {m},{k},{n})"
        );
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let device = self.shared.device.clone();
        let kern = self.kernel_spv("gemm_coopmat", gemm_spv(), 3, 12);

        let a16: Vec<u16> = a.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let b16: Vec<u16> = b.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
        let buf_a = self.alloc(a16.len() * 2, BufferUsage::Staging)?;
        let buf_b = self.alloc(b16.len() * 2, BufferUsage::Staging)?;
        let buf_c = self.alloc(m * n * 4, BufferUsage::Readback)?;
        self.upload(buf_a.as_ref(), bytemuck::cast_slice(&a16))?;
        self.upload(buf_b.as_ref(), bytemuck::cast_slice(&b16))?;

        unsafe {
            device
                .reset_descriptor_pool(kern.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| be(format!("reset pool: {e}")))?;
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(kern.desc_pool)
                        .set_layouts(std::slice::from_ref(&kern.ds_layout)),
                )
                .map_err(|e| be(format!("alloc set: {e}")))?[0]
        };
        let vk_a = unsafe { as_vk_buf(buf_a.as_ref()) }.buffer;
        let vk_b = unsafe { as_vk_buf(buf_b.as_ref()) }.buffer;
        let vk_c = unsafe { as_vk_buf(buf_c.as_ref()) }.buffer;
        let infos = [
            vk::DescriptorBufferInfo {
                buffer: vk_a,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            vk::DescriptorBufferInfo {
                buffer: vk_c,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i..i + 1])
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u8; 12];
        push[0..4].copy_from_slice(&(m as u32).to_ne_bytes());
        push[4..8].copy_from_slice(&(n as u32).to_ne_bytes());
        push[8..12].copy_from_slice(&(k as u32).to_ne_bytes());

        let gx = (n / 16) as u32;
        let gy = (m / 16) as u32;
        let shared = std::sync::Arc::clone(&self.shared);
        self.one_shot(move |cmd| unsafe {
            shared
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kern.pipeline);
            shared.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kern.pipeline_layout,
                0,
                &[set],
                &[],
            );
            shared.device.cmd_push_constants(
                cmd,
                kern.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            shared.device.cmd_dispatch(cmd, gx, gy, 1);
        })?;

        let mut c_bytes = vec![0u8; m * n * 4];
        self.download(buf_c.as_ref(), &mut c_bytes)?;
        Ok(bytemuck::cast_slice(&c_bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    #[ignore = "requires a Vulkan GPU with cooperative matrix"]
    fn coopmat_gemm_matches_cpu() {
        let be = VulkanBackend::new().unwrap();
        let (m, k, n) = (64usize, 48usize, 32usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let got = be.matmul_f16(&a, &b, m, k, n).unwrap();
        let want = cpu(&a, &b, m, k, n);
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let denom = w.abs().max(1.0);
            max_rel = max_rel.max((g - w).abs() / denom);
        }
        println!("coopmat GEMM max_rel_err = {max_rel:.4e}");
        assert!(max_rel < 2e-2, "coopmat GEMM rel err {max_rel} too high");
    }

    #[test]
    #[ignore = "benchmark, requires GPU"]
    fn coopmat_gemm_bench() {
        let be = VulkanBackend::new().unwrap();
        let s = 2048usize; // 2048^3
        let a = vec![0.01f32; s * s];
        let b = vec![0.02f32; s * s];
        let _ = be.matmul_f16(&a, &b, s, s, s).unwrap(); // warm
        let t = std::time::Instant::now();
        let iters = 5;
        for _ in 0..iters {
            let _ = be.matmul_f16(&a, &b, s, s, s).unwrap();
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let flops = 2.0 * (s as f64).powi(3);
        println!(
            "coopmat GEMM {s}^3: {:.2} ms, {:.1} GFLOP/s",
            dt * 1e3,
            flops / dt / 1e9
        );
    }
}
