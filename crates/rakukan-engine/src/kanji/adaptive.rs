//! Deterministic, per-decoding-width adaptive inference policy.
pub(super) fn enabled(opt_in: bool, gpu_layers: u32) -> bool {
    opt_in && gpu_layers > 0
}

const GPU_SIDE_CPU_PROBE_INTERVAL: u8 = 50;

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
    pub fn next(self) -> Device {
        // Obtain a CPU baseline after three GPU conversions, then refresh it
        // rarely: CPU speed barely changes, and every CPU probe is a slow
        // conversion for the user. While on CPU, probe the GPU often so we
        // return quickly once it frees up. Probes serve the request; never run
        // inference twice.
        let interval = match (self.preferred, self.cpu_ms) {
            (Device::Gpu, None) => 3,
            (Device::Gpu, Some(_)) => GPU_SIDE_CPU_PROBE_INTERVAL,
            (Device::Cpu, _) => 8,
        };
        if self.since_probe >= interval {
            match self.preferred {
                Device::Gpu => Device::Cpu,
                Device::Cpu => Device::Gpu,
            }
        } else {
            self.preferred
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
        if let (Some(gpu), Some(cpu)) = (self.gpu_ms, self.cpu_ms) {
            let switch = match self.preferred {
                Device::Gpu => gpu > cpu * 1.30,
                // Only GPU probes count as recovery evidence.
                Device::Cpu => gpu < cpu * 0.85,
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
    fn calibrates_and_probes_without_duplicate_inference() {
        let mut p = Policy::default();
        for _ in 0..3 {
            assert_eq!(p.next(), Device::Gpu);
            p = p.observe(Device::Gpu, 100.0, 10);
        }
        assert_eq!(p.next(), Device::Cpu);
        p = p.observe(Device::Cpu, 200.0, 10);
        for _ in 0..GPU_SIDE_CPU_PROBE_INTERVAL {
            assert_eq!(p.next(), Device::Gpu);
            p = p.observe(Device::Gpu, 100.0, 10);
        }
        assert_eq!(p.next(), Device::Cpu);
    }

    #[test]
    fn slow_gpu_requires_evidence_and_recovery_requires_probes() {
        let mut p = Policy::default().observe(Device::Cpu, 100.0, 10);
        p = p.observe(Device::Gpu, 200.0, 10);
        assert_eq!(p.preferred, Device::Gpu);
        p = p.observe(Device::Gpu, 200.0, 10);
        assert_eq!(p.preferred, Device::Cpu);
        for _ in 0..8 {
            assert_eq!(p.next(), Device::Cpu);
            p = p.observe(Device::Cpu, 100.0, 10);
        }
        assert_eq!(p.next(), Device::Gpu);
        p = p.observe(Device::Gpu, 10.0, 10);
        p = p.observe(Device::Gpu, 10.0, 10);
        assert_eq!(p.preferred, Device::Cpu);
        for _ in 0..8 {
            p = p.observe(Device::Cpu, 100.0, 10);
        }
        assert_eq!(p.preferred, Device::Cpu);
        p = p.observe(Device::Gpu, 10.0, 10);
        assert_eq!(p.preferred, Device::Gpu);
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
