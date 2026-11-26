//! nuts-rs driver for Mojo models exported through a small C ABI.
//!
//! A model library `lib<name>.dylib` exports:
//!   int32_t <name>_dim(const double* data)
//!   int32_t <name>_logp_grad(const double* data, const double* x, double* grad, double* logp)
//! `logp_grad` returns 0 on success and nonzero if the point is invalid (recoverable).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use nuts_rs::{
    ArrowConfig, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, LogpError, Model, Sampler,
    SamplerWaitResult,
};
use nuts_storable::HasDims;
use rand::{Rng, RngExt};
use serde::Serialize;
use thiserror::Error;

type DimFn = unsafe extern "C" fn(*const f64) -> i32;
type LogpGradFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64, *mut f64) -> i32;

/// A loaded Mojo model: the library handle, the two symbols and the model data buffer.
pub struct MojoLib {
    _lib: libloading::Library,
    logp_grad: LogpGradFn,
    data: Vec<f64>,
    dim: usize,
}

impl MojoLib {
    /// Load `path`, resolve `<name>_dim` and `<name>_logp_grad`, and bind `data`.
    pub fn load(path: &str, name: &str, data: Vec<f64>) -> Result<Self> {
        // SAFETY: loading a library we built ourselves; symbols have the documented ABI.
        unsafe {
            let lib = libloading::Library::new(path)?;
            let dim_fn: DimFn = *lib.get::<DimFn>(format!("{name}_dim").as_bytes())?;
            let logp_grad: LogpGradFn =
                *lib.get::<LogpGradFn>(format!("{name}_logp_grad").as_bytes())?;
            let dptr = if data.is_empty() {
                std::ptr::null()
            } else {
                data.as_ptr()
            };
            let dim = dim_fn(dptr);
            if dim <= 0 {
                return Err(anyhow!("{name}_dim returned {dim}"));
            }
            Ok(Self {
                _lib: lib,
                logp_grad,
                data,
                dim: dim as usize,
            })
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate logp and gradient at `x`. Returns (status, logp).
    pub fn eval(&self, x: &[f64], grad: &mut [f64]) -> (i32, f64) {
        assert_eq!(x.len(), self.dim);
        assert_eq!(grad.len(), self.dim);
        let mut lp = 0.0f64;
        let dptr = if self.data.is_empty() {
            std::ptr::null()
        } else {
            self.data.as_ptr()
        };
        // SAFETY: buffers have length dim as checked above; data outlives the call.
        let rc = unsafe { (self.logp_grad)(dptr, x.as_ptr(), grad.as_mut_ptr(), &mut lp) };
        (rc, lp)
    }
}

#[derive(Debug, Error)]
pub enum MojoLogpError {
    #[error("model returned status {0} (non-finite logp)")]
    BadPoint(i32),
}

impl LogpError for MojoLogpError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// Per-chain logp function handed to nuts-rs.
///
/// Gradient evaluations are not counted here: a shared atomic counter across chains
/// caused measurable cache-line contention (the kernel is ~33 ns). Use the `n_steps`
/// sampler statistic instead, which is also what nutpie reports.
#[derive(Clone)]
pub struct MojoLogp {
    lib: Arc<MojoLib>,
}

impl HasDims for MojoLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), self.lib.dim as u64),
            ("dim".to_string(), self.lib.dim as u64),
        ])
    }
}

impl CpuLogpFunc for MojoLogp {
    type LogpError = MojoLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.lib.dim
    }

    fn logp(&mut self, x: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        let (rc, lp) = self.lib.eval(x, grad);
        if rc != 0 {
            return Err(MojoLogpError::BadPoint(rc));
        }
        Ok(lp)
    }

    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, CpuMathError> {
        Ok(array.to_vec())
    }
}

struct MojoModel {
    math: CpuMath<MojoLogp>,
}

impl Model for MojoModel {
    type Math<'model>
        = CpuMath<MojoLogp>
    where
        Self: 'model;

    fn math<R: Rng + ?Sized>(&self, _rng: &mut R) -> Result<Self::Math<'_>> {
        Ok(self.math.clone())
    }

    fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
        for p in position.iter_mut() {
            *p = rng.random_range(-2.0..2.0);
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub struct ChainOut {
    /// Post-warmup draws, row-major (draws, dim), unconstrained space.
    pub draws: Vec<f64>,
    pub n_draws: usize,
    pub divergences: u64,
    /// Leapfrog steps summed over post-warmup draws (one gradient each).
    pub n_steps_draws: u64,
    /// Leapfrog steps summed over warmup and draws.
    pub n_steps_total: u64,
    pub step_size: f64,
}

#[derive(Serialize)]
pub struct RunOut {
    pub dim: usize,
    pub chains: Vec<ChainOut>,
    /// Wall time from Sampler::new to having all traces, seconds.
    pub sample_seconds: f64,
    /// Leapfrog steps (= gradient evaluations) across chains, warmup + draws.
    pub grad_evals: u64,
    pub settings: serde_json::Value,
    /// Sampling wall times of all in-process repeats (first entry is this run).
    pub repeat_seconds: Vec<f64>,
}

/// Settings matching nutpie 0.16.11 `PyNutsSettings.Diag` defaults (checked via `as_dict()`).
pub fn nutpie_like_settings(chains: usize, tune: u64, draws: u64, seed: u64) -> DiagNutsSettings {
    let mut s = DiagNutsSettings {
        num_chains: chains,
        num_tune: tune,
        num_draws: draws,
        seed,
        maxdepth: 10,
        max_energy_error: 1000.0,
        ..Default::default()
    };
    s.adapt_options.step_size_settings.target_accept = 0.8;
    s
}

/// Read a column as flat f64 values (scalar columns, or list columns flattened row-major).
fn get_f64(batch: &RecordBatch, k: &str) -> Result<Vec<f64>> {
    let col = batch.column_by_name(k).ok_or_else(|| {
        let names: Vec<_> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        anyhow!("missing column {k}; have {names:?}")
    })?;
    let values = match col.data_type() {
        DataType::List(_) => col.as_list::<i32>().values().clone(),
        DataType::LargeList(_) => col.as_list::<i64>().values().clone(),
        DataType::FixedSizeList(_, _) => col.as_fixed_size_list().values().clone(),
        _ => col.clone(),
    };
    let f = cast(&values, &DataType::Float64)?;
    Ok(f.as_primitive::<Float64Type>().values().to_vec())
}

/// Run NUTS with `cores` worker threads and return draws plus summary stats.
pub fn run(lib: Arc<MojoLib>, settings: DiagNutsSettings, cores: usize) -> Result<RunOut> {
    let dim = lib.dim;
    let model = MojoModel {
        math: CpuMath::new(MojoLogp { lib }),
    };
    let settings_json = serde_json::to_value(settings)?;
    let start = Instant::now();
    let mut sampler = Some(Sampler::new(
        model,
        settings,
        ArrowConfig::default(),
        cores,
        None,
    )?);
    let traces = loop {
        let s = sampler.take().expect("sampler present");
        match s.wait_timeout(Duration::from_secs(3600)) {
            SamplerWaitResult::Trace(t) => break t,
            SamplerWaitResult::Timeout(s) => sampler = Some(s),
            SamplerWaitResult::Err(e, _) => return Err(e),
        }
    };
    let sample_seconds = start.elapsed().as_secs_f64();

    let mut chains = Vec::new();
    for tr in traces.iter() {
        let st = &tr.sample_stats;
        let tuning = get_f64(st, "tuning")?;
        let diverging = get_f64(st, "diverging")?;
        let n_steps = get_f64(st, "n_steps")?;
        let step = get_f64(st, "step_size")?;
        let draws = get_f64(&tr.posterior, "value")
            .or_else(|_| get_f64(&tr.posterior, "unconstrained_parameter"))?;
        let total = tuning.len();
        if draws.len() != total * dim {
            return Err(anyhow!("draws len {} != {} x {}", draws.len(), total, dim));
        }
        let keep: Vec<usize> = (0..total).filter(|&i| tuning[i] == 0.0).collect();
        let mut out = Vec::with_capacity(keep.len() * dim);
        for &i in &keep {
            out.extend_from_slice(&draws[i * dim..(i + 1) * dim]);
        }
        chains.push(ChainOut {
            n_draws: keep.len(),
            divergences: keep.iter().map(|&i| diverging[i] as u64).sum(),
            n_steps_draws: keep.iter().map(|&i| n_steps[i] as u64).sum(),
            n_steps_total: n_steps.iter().map(|&x| x as u64).sum(),
            step_size: keep.last().map(|&i| step[i]).unwrap_or(f64::NAN),
            draws: out,
        });
    }
    let grad_evals = chains.iter().map(|c| c.n_steps_total).sum();
    Ok(RunOut {
        dim,
        chains,
        sample_seconds,
        grad_evals,
        settings: settings_json,
        repeat_seconds: vec![sample_seconds],
    })
}
