//! Deterministic, per-decoding-width adaptive inference policy.
pub(super) fn enabled(opt_in: bool, gpu_layers: u32) -> bool {
    opt_in && gpu_layers > 0
}

// A healthy GPU runs at ~10-20 ms per reading char. Below this the GPU is
// never worth leaving, so no CPU probe is spent: a CPU probe is a real user
// conversion, and under CPU contention it has taken 8-30 s (2026-09-19).
const GPU_SLOW_MS_PER_CHAR: f64 = 40.0;
// While the GPU is slow, refresh the CPU baseline every N GPU conversions.
const SLOW_GPU_CPU_PROBE_INTERVAL: u8 = 8;
// While on CPU, probe the GPU often so we return quickly once it frees up.
const CPU_SIDE_GPU_PROBE_INTERVAL: u8 = 4;
// A CPU conversion this slow means the CPU is contended; stop using it at once
// instead of waiting for averaged evidence.
const CPU_BAIL_MS_PER_CHAR: f64 = 300.0;
const CPU_BAIL_TOTAL_MS: f64 = 3000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Device {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Policy {
    pub preferred: Device,
    pub gpu_ms: Option<f64>,
    pub cpu_ms: Option<f64>,
    since_probe: u8,
    evidence: u8,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            preferred: Device::Gpu,
            gpu_ms: None,
            cpu_ms: None,
            since_probe: 0,
            evidence: 0,
        }
    }
}

impl Policy {
    fn gpu_is_slow(&self) -> bool {
        self.gpu_ms.is_some_and(|g| g >= GPU_SLOW_MS_PER_CHAR)
    }

    pub fn next(self) -> Device {
        // Probes serve the request; never run inference twice.
        match self.preferred {
            Device::Gpu => {
                // Touch the CPU only once the GPU itself is measurably slow.
                let interval = if self.cpu_ms.is_none() {
                    1
                } else {
                    SLOW_GPU_CPU_PROBE_INTERVAL
                };
                if self.gpu_is_slow() && self.since_probe >= interval {
                    Device::Cpu
                } else {
                    Device::Gpu
                }
            }
            Device::Cpu => {
                if self.since_probe >= CPU_SIDE_GPU_PROBE_INTERVAL {
                    Device::Gpu
                } else {
                    Device::Cpu
                }
            }
        }
    }

    /// Pure state transition. Invalid/empty measurements cannot bias selection.
    pub fn observe(mut self, device: Device, elapsed_ms: f64, chars: usize) -> Self {
        if chars == 0 || !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
            return self;
        }
        let sample = elapsed_ms / chars as f64;
        let average = match device {
            Device::Gpu => &mut self.gpu_ms,
            Device::Cpu => &mut self.cpu_ms,
        };
        *average = Some(average.map_or(sample, |old| old * 0.5 + sample * 0.5));
        if device == self.preferred {
            self.since_probe = self.since_probe.saturating_add(1);
        } else {
            self.since_probe = 0;
        }
        if device == Device::Cpu
            && self.preferred == Device::Cpu
            && (sample >= CPU_BAIL_MS_PER_CHAR || elapsed_ms >= CPU_BAIL_TOTAL_MS)
        {
            self.preferred = Device::Gpu;
            self.evidence = 0;
            self.since_probe = 0;
            return self;
        }
        if let (Some(gpu), Some(cpu)) = (self.gpu_ms, self.cpu_ms) {
            let switch = match self.preferred {
                Device::Gpu => self.gpu_is_slow() && gpu > cpu * 1.30,
                // Only GPU probes count as recovery evidence.
                Device::Cpu => !self.gpu_is_slow() || gpu < cpu * 0.85,
            };
            if !switch {
                self.evidence = 0;
            } else if self.preferred == Device::Gpu || device == Device::Gpu {
                self.evidence += 1;
                if self.evidence >= 2 {
                    self.preferred = match self.preferred {
                        Device::Gpu => Device::Cpu,
                        Device::Cpu => Device::Gpu,
                    };
                    self.evidence = 0;
                    self.since_probe = 0;
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_and_gpu_layers_are_both_required() {
        assert!(!enabled(false, 16));
        assert!(!enabled(false, 0));
        assert!(!enabled(true, 0));
        assert!(enabled(true, 16));
    }

    #[test]
    fn healthy_gpu_never_probes_cpu() {
        let mut p = Policy::default();
        for _ in 0..200 {
            assert_eq!(p.next(), Device::Gpu);
            p = p.observe(Device::Gpu, 150.0, 10);
        }
        assert_eq!(p.cpu_ms, None);
    }

    #[test]
    fn slow_gpu_requires_evidence_and_recovery_requires_probes() {
        let mut p = Policy::default().observe(Device::Gpu, 500.0, 10);
        assert_eq!(p.next(), Device::Cpu);
        p = p.observe(Device::Cpu, 300.0, 10);
        assert_eq!(p.preferred, Device::Gpu);
        p = p.observe(Device::Gpu, 500.0, 10);
        assert_eq!(p.preferred, Device::Cpu);
        for _ in 0..CPU_SIDE_GPU_PROBE_INTERVAL {
            assert_eq!(p.next(), Device::Cpu);
            p = p.observe(Device::Cpu, 300.0, 10);
        }
        assert_eq!(p.next(), Device::Gpu);
        p = p.observe(Device::Gpu, 100.0, 10);
        assert_eq!(p.preferred, Device::Cpu);
        p = p.observe(Device::Gpu, 100.0, 10);
        assert_eq!(p.preferred, Device::Gpu);
    }

    #[test]
    fn contended_cpu_conversion_returns_to_gpu_immediately() {
        let mut p = Policy::default().observe(Device::Gpu, 500.0, 10);
        p = p.observe(Device::Cpu, 300.0, 10);
        p = p.observe(Device::Gpu, 500.0, 10);
        assert_eq!(p.preferred, Device::Cpu);
        p = p.observe(Device::Cpu, 8285.0, 4);
        assert_eq!(p.preferred, Device::Gpu);
        assert_eq!(p.next(), Device::Gpu);
    }

    #[test]
    fn normalization_dead_band_and_invalid_samples() {
        let p = Policy::default().observe(Device::Cpu, 100.0, 10);
        let mut p = p.observe(Device::Gpu, 240.0, 20);
        for _ in 0..10 {
            p = p.observe(Device::Gpu, 120.0, 10);
        }
        assert_eq!(p.preferred, Device::Gpu);
        let original = p.gpu_ms;
        for (ms, chars) in [(1.0, 0), (f64::NAN, 10), (0.0, 10), (-1.0, 10)] {
            p = p.observe(Device::Gpu, ms, chars);
        }
        assert_eq!(p.gpu_ms, original);
    }
}
