"""Paired per-seed statistics for the key contrasts (same walks across conditions => paired)."""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/results_gt"
W = 50  # average mass over a +-W token window around each probe length


def window_means(curves, ts, w=W):
    M = np.array(curves, dtype=float)              # [seeds, T-1]
    return np.stack([M[:, max(0, t - w):t + w].mean(1) for t in ts], 1)   # [seeds, len(ts)]


def paired(a, b, ts, label):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    A, B = window_means(a, ts), window_means(b, ts)
    D = A - B
    out = []
    for i, t in enumerate(ts):
        d = D[:, i]
        tt, p = stats.ttest_1samp(d, 0) if len(d) > 1 else (np.nan, np.nan)
        out.append(f"t={t}: Δ={d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(len(d)):.3f} (p={p:.3g}, {int((d>0).sum())}/{len(d)} seeds >0)")
    print(f"{label:58s} | " + " | ".join(out))


ts = (100, 200, 400)
if os.path.exists(f"{RES}/e2.json"):
    e2 = json.load(open(f"{RES}/e2.json")); D = e2["dense"]
    print(f"== E2 (n={len(D['none'])} seeds) contrasts vs baseline, mass averaged over ±{W} tokens ==")
    for k in ("native_gt2_b2.0", "native_gt2_b3.0", "native_gt2_calB_b2.0", "extracted_b2.0", "random_gt2_b2.0_plane123"):
        if k in D: paired(D[k], D["none"], ts, f"E2 {k} - none")
    ov = e2["override"]
    for b in ("b2.0", "b3.0"):
        paired(ov[b]["injected_map"], ov["none_injmap"], ts, f"E2 override {b}: injected-map - chance")
        paired(ov[b]["true_map"], D["none"], ts, f"E2 override {b}: true-map - baseline")
    rows = e2["unseen"]
    for c in ("ctx+query_trueX", "ctx+query_claimY", "query_only_trueX"):
        dx = np.array([r["conds"][c]["P_nbrs_X"] - r["conds"]["none"]["P_nbrs_X"] for r in rows])
        dy = np.array([r["conds"][c]["P_nbrs_Y"] - r["conds"]["none"]["P_nbrs_Y"] for r in rows])
        print(f"E2 unseen {c:20s} ΔP(nbrs X)={dx.mean():+.3f}±{dx.std(ddof=1)/np.sqrt(len(dx)):.3f} (p={stats.ttest_1samp(dx,0).pvalue:.3g})  "
              f"ΔP(nbrs Y)={dy.mean():+.3f}±{dy.std(ddof=1)/np.sqrt(len(dy)):.3f} (p={stats.ttest_1samp(dy,0).pvalue:.3g})  n={len(rows)}")
if os.path.exists(f"{RES}/e2b.json"):
    eb = json.load(open(f"{RES}/e2b.json")); Db = eb["dense"]
    print(f"\n== E2b (n={len(Db['self_gt2_b2.0'])} seeds) ==")
    for k in ("self_gt2_b1.0", "self_gt2_b2.0", "self_gt2_b3.0", "self_gt3_b2.0", "analytic_inE_gt2_b2.0",
              f"analytic_hs{eb['meta']['h_main']}E_gt2_b2.0", "shared_top2_gt2_b2.0", "shared_rand0_gt2_b2.0", "shared_rand1_gt2_b2.0"):
        if k in Db: paired(Db[k], Db["none"][:len(Db[k])], ts, f"E2b {k} - none")
    if "extracted_b2.0" in Db:
        paired(Db["self_gt2_b2.0"], Db["extracted_b2.0"][:len(Db["self_gt2_b2.0"])], ts, "E2b self_gt2_b2.0 - extracted_b2.0")
    ov = eb.get("override", {})
    e2 = json.load(open(f"{RES}/e2.json")) if os.path.exists(f"{RES}/e2.json") else None
    for b in ("self_b2.0", "self_b3.0"):
        if b in ov:
            paired(ov[b]["injected_map"], e2["override"]["none_injmap"][:len(ov[b]["injected_map"])], ts, f"E2b override {b}: injected-map - chance")
            paired(ov[b]["true_map"], Db["none"][:len(ov[b]["true_map"])], ts, f"E2b override {b}: true-map - baseline")
    rows = eb["unseen"]
    for c in ("self_trueX", "self_claimY", "self_queryOnly"):
        dx = np.array([r["conds"][c]["P_nbrs_X"] - r["conds"]["none"]["P_nbrs_X"] for r in rows])
        dy = np.array([r["conds"][c]["P_nbrs_Y"] - r["conds"]["none"]["P_nbrs_Y"] for r in rows])
        print(f"E2b unseen {c:16s} ΔP(nbrs X)={dx.mean():+.3f}±{dx.std(ddof=1)/np.sqrt(len(dx)):.3f} (p={stats.ttest_1samp(dx,0).pvalue:.3g})  "
              f"ΔP(nbrs Y)={dy.mean():+.3f}±{dy.std(ddof=1)/np.sqrt(len(dy)):.3f} (p={stats.ttest_1samp(dy,0).pvalue:.3g})  n={len(rows)}")
if os.path.exists(f"{RES}/e3.json"):
    e3 = json.load(open(f"{RES}/e3.json")); C = e3["curves"]
    q = lambda n: [ps[-1] for ps in C[n]]
    print(f"\n== E3 (n={len(C['none_demos0'])} seeds), query grid ==")
    ts3 = (100, 200, 400)
    for n in range(4):
        paired(q(f"native_demos{n}"), q(f"none_demos{n}"), ts3, f"E3 native_demos{n} - none_demos{n}")
    paired(q("native_demos3"), q("native_demos3_queryOnly"), ts3, "E3 injected demos - uninjected demos (query injected)")
    paired(q("native_demos3"), q("native_demos0"), ts3, "E3 native 3 demos - native 0 demos")
    paired(q("none_demos3"), q("none_demos0"), ts3, "E3 none 3 demos - none 0 demos (task-format effect)")
    paired(q("random_demos3"), q("none_demos3"), ts3, "E3 random 3 demos - none 3 demos")
    paired(q("native_demos3_demoScrambled"), q("native_demos3"), ts3, "E3 scrambled demos - consistent demos")
