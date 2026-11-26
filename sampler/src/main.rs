//! CLI: mojo-nuts --lib PATH --model NAME [--data FILE.f64] [--chains 4] [--tune 1000]
//!                [--draws 1000] [--seed 1] [--cores N] [--out out.json]
use std::sync::Arc;

use anyhow::{Result, anyhow};
use mojo_nuts::{MojoLib, nutpie_like_settings, run};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| -> Option<String> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let lib_path = get("--lib").ok_or_else(|| anyhow!("--lib required"))?;
    let model = get("--model").ok_or_else(|| anyhow!("--model required"))?;
    let chains: usize = get("--chains").map_or(Ok(4), |v| v.parse())?;
    let tune: u64 = get("--tune").map_or(Ok(1000), |v| v.parse())?;
    let draws: u64 = get("--draws").map_or(Ok(1000), |v| v.parse())?;
    let seed: u64 = get("--seed").map_or(Ok(1), |v| v.parse())?;
    let cores: usize = get("--cores").map_or(Ok(chains), |v| v.parse())?;
    let data = match get("--data") {
        Some(p) => std::fs::read(p)?
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect(),
        None => Vec::new(),
    };
    let lib = Arc::new(MojoLib::load(&lib_path, &model, data)?);
    if let Some(n) = get("--bench-evals") {
        // Kernel-only timing from Rust: logp+grad at x = 0.1 repeated n times.
        let n: usize = n.parse()?;
        let x = vec![0.1f64; lib.dim()];
        let mut g = vec![0.0f64; lib.dim()];
        let mut acc = 0.0;
        for _ in 0..n.min(1000) {
            acc += lib.eval(&x, &mut g).1;
        }
        let t = std::time::Instant::now();
        for _ in 0..n {
            acc += lib.eval(std::hint::black_box(&x), &mut g).1;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / n as f64;
        println!("{{\"ns_per_eval\": {ns}, \"checksum\": {acc}}}");
        return Ok(());
    }
    let repeat: usize = get("--repeat").map_or(Ok(1), |v| v.parse())?;
    // Repeat in-process (seed + i) to separate warm-process timing from process start-up.
    let mut times = Vec::new();
    let mut out = None;
    for i in 0..repeat {
        let settings = nutpie_like_settings(chains, tune, draws, seed + i as u64);
        let r = run(lib.clone(), settings, cores)?;
        times.push(r.sample_seconds);
        if i == 0 {
            out = Some(r);
        }
    }
    let mut out = out.ok_or_else(|| anyhow!("--repeat must be >= 1"))?;
    out.repeat_seconds = times;
    let json = serde_json::to_string(&out)?;
    match get("--out") {
        Some(p) => std::fs::write(p, json)?,
        None => println!("{json}"),
    }
    eprintln!(
        "sampled {} chains in {:.4}s, {} grad evals",
        out.chains.len(),
        out.sample_seconds,
        out.grad_evals
    );
    Ok(())
}
