//! Integration test: load the Mojo eight-schools library and sample a short run.
use std::path::PathBuf;
use std::sync::Arc;

use mojo_nuts::{MojoLib, nutpie_like_settings, run};

fn lib_path() -> Option<String> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../mojo_capi/libes8.dylib");
    p.exists().then(|| p.to_string_lossy().into_owned())
}

#[test]
fn es8_gradient_matches_finite_differences() {
    let Some(path) = lib_path() else { return };
    let lib = MojoLib::load(&path, "es8", Vec::new()).expect("load");
    assert_eq!(lib.dim(), 10);
    let x: Vec<f64> = (0..10).map(|i| 0.1 * i as f64 - 0.3).collect();
    let mut g = vec![0.0; 10];
    let (rc, _) = lib.eval(&x, &mut g);
    assert_eq!(rc, 0);
    let h = 1e-6;
    let mut tmp = vec![0.0; 10];
    for i in 0..10 {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[i] += h;
        xm[i] -= h;
        let fd = (lib.eval(&xp, &mut tmp).1 - lib.eval(&xm, &mut tmp).1) / (2.0 * h);
        assert!(
            (fd - g[i]).abs() < 1e-5 * (1.0 + g[i].abs()),
            "i={i} fd={fd} g={}",
            g[i]
        );
    }
}

#[test]
fn es8_short_run_produces_draws() {
    let Some(path) = lib_path() else { return };
    let lib = Arc::new(MojoLib::load(&path, "es8", Vec::new()).expect("load"));
    let out = run(lib, nutpie_like_settings(2, 200, 100, 3), 2).expect("run");
    assert_eq!(out.chains.len(), 2);
    for c in &out.chains {
        assert_eq!(c.n_draws, 100);
        assert_eq!(c.draws.len(), 100 * 10);
        assert!(c.draws.iter().all(|v| v.is_finite()));
    }
    assert!(out.grad_evals > 0);
}
