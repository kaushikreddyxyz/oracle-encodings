#!/usr/bin/env python3
"""Tables + figures for the graft-code campaign, intact prompt (results/runs/<tag>/<run>/eval/{metrics.json,probe_walls.json}).

  fig_codes_curves.{pdf,svg}   p(SP) and solved vs step, true + phantom world, one line per run (+ 5-seed base / edge refs)
  fig_codes_probes.{pdf,svg}   wall-probe accuracy vs layer (absolute walls at path tokens / at PATH_START, local N/E/S/W)
  codes_final.csv / codes_probes.csv / tables.md
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; CUR = HERE.parent / "curves"; RES = HERE / "results" / "runs"; G = 6
BASE, EDGE = "#2a7a57", "#000c7a"
PAL = {"vertex": "#eb6834", "degree": "#898781", "ego": "#1c5cab", "onion": "#a3369c", "edge": EDGE, "axdeg": "#2a7a57", "axdeg1h": "#eb6834"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "legend.frameon": False})
STEPS_PER_EPOCH = 7812


def runs():
    out = {}
    for d in sorted(RES.glob(f"g{G}-s*")):
        for r in list(d.glob("fast-inj-*")) + list(d.glob("fast-baseline*")):
            if (r / "eval" / "metrics.json").exists():
                parts = d.name.split("-")            # g6-s0-<code>-e<E>-oe  (intact prompt, start = origin_end)
                code = parts[2]; kind = "fixed" if "inj-fixed" in r.name else "trainable" if "inj-trainable" in r.name else "none"
                out[f"{code} {kind}" if code != "base" else "base (no graft)"] = dict(dir=r, code=code, kind=kind, dummy=False)
    return out


def curves(r):
    rows = json.load(open(r["dir"] / "eval" / "metrics.json"))["rows"]
    n_it = max(m["iteration"] for m in rows)
    off = STEPS_PER_EPOCH if r.get("cont") else 0     # continuation runs restart their step counter after epoch 1
    out = {}
    for m in rows:
        x = off + ((n_it + 12) if m["iteration"] == -1 else m["iteration"])
        out.setdefault(m["world"], []).append((x, m["p_sp"], m["solved"]))
    return {w: np.array(sorted(v)) for w, v in out.items()}


def ref_rows():
    """all_metrics.csv of the curves campaign: current location, tables/, or the committed copy (the paper tree is being reorganised)."""
    import io, subprocess
    for p in (CUR / "all_metrics.csv", CUR / "tables" / "all_metrics.csv"):
        if p.exists():
            return csv.DictReader(open(p))
    txt = subprocess.run(["git", "-C", str(HERE), "show", "HEAD:maze/paper/curves/all_metrics.csv"], capture_output=True, text=True, check=True).stdout
    return csv.DictReader(io.StringIO(txt))


def ref_final():
    acc = {}
    for row in ref_rows():
        if int(row["grid"]) == G and row["final"] == "True":
            acc.setdefault((row["cond"], row["world"]), []).append((float(row["p_sp"]), float(row["solved"])))
    return {k: np.mean(v, 0) for k, v in acc.items()}


def md_table(rows, cols, fmt="{:.3f}"):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def intact_rows(R):
    """Intact-prompt runs: standard set (route-changed items) and one-swap set, vs the 5-seed edge references."""
    RES2 = HERE / "results"
    S = np.load(CUR / "results/distance/sets" / f"g{G}.npz"); rc = (~S["path_same"]) & (S["level"] > 0); same = S["level"] == 0
    Lc = np.load(CUR / "results/local/sets" / f"g{G}.npz")
    drows, lrows = [], []
    def dist(label, files):
        a = [np.load(f) for f in files]
        if a: drows.append(dict(model=label, n=len(a), same_maze=float(np.mean([x["solved_graft"][same].mean() for x in a])),
                                route_changed_graft=float(np.mean([x["solved_graft"][rc].mean() for x in a])),
                                route_changed_prompt=float(np.mean([x["solved_token"][rc].mean() for x in a]))))
    def loc(label, files):
        a = [np.load(f) for f in files]
        if not a: return
        vals = []
        for z in a:
            N = len(z["tid_a"]); m = Lc["family"][:N] == 0; g = lambda c, k: z[f"{c}__{k}"][:N]
            vals.append(dict(p_new=float(np.exp(g("graft_B", "lp_b"))[m].mean()), top1_new=float((g("graft_B", "argmax") == z["tid_b"][:N])[m].mean()),
                             gain_nats=float((g("graft_B", "lp_b") - g("graft_A", "lp_b"))[m].mean()), used_door=float(g("graft_B", "used")[m].mean()),
                             crossed_wall=float(g("graft_B", "crossed")[m].mean()), solved_new=float(g("graft_B", "solved_b")[m].mean()),
                             prompt_only_p_new=float(np.exp(g("prompt_B", "lp_b"))[m].mean()), both_p_new=float(np.exp(g("both_B", "lp_b"))[m].mean())))
        lrows.append(dict(model=label, n=len(a), **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}))
    for name, r in R.items():
        if r["dummy"] or r.get("cont"): continue
        tag = r["dir"].parent.name
        dist(name, list((RES2 / "dist_out").glob(f"g{G}__{r['kind']}__{tag}.npz"))); loc(name, list((RES2 / "local_out").glob(f"g{G}__{r['kind']}__{tag}.npz")))
    for kind in ("fixed", "trainable"):
        dist(f"REF edge {kind} (5 seeds, 1 ep, adjlist_end)", sorted((CUR / "results/distance/out").glob(f"g{G}__{kind}__g{G}-s?.npz")))
        loc(f"REF edge {kind} (5 seeds, 1 ep, adjlist_end)", sorted((CUR / "results/local/out").glob(f"g{G}__{kind}__g{G}-s?.npz")))
    return drows, lrows


def ablate_rows(R):
    rows = []
    for name, r in R.items():
        f = r["dir"] / "eval" / "ablate.json"
        if f.exists():
            for x in json.load(open(f))["rows"]:
                rows.append(dict(model=name, **{k: v for k, v in x.items()}))
    return rows


def main():
    R = runs(); ref = ref_final()
    rows = []
    for name, r in R.items():
        c = curves(r)
        for w, arr in c.items():
            near1 = arr[np.argmin(np.abs(arr[:, 0] - STEPS_PER_EPOCH))] if not r.get("cont") else np.array([np.nan] * 3)
            rows.append(dict(model=name, world=w, p_sp_1ep=near1[1], solved_1ep=near1[2], p_sp_final=arr[-1, 1], solved_final=arr[-1, 2], final_step=int(arr[-1, 0])))
    for (cond, w), (p, s) in sorted(ref.items()):
        rows.append(dict(model=f"REF {cond} (5 seeds, intact prompt)", world=w, p_sp_1ep=p, solved_1ep=s, p_sp_final=float("nan"), solved_final=float("nan"), final_step=STEPS_PER_EPOCH))
    with open(HERE / "codes_final.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    md = ["| model | world | p(SP) 1 ep | solved 1 ep | p(SP) final | solved final | final step |", "|---|---|---|---|---|---|---|"]
    md += [f"| {x['model']} | {x['world']} | {x['p_sp_1ep']:.3f} | {x['solved_1ep']:.3f} | {x['p_sp_final']:.3f} | {x['solved_final']:.3f} | {x['final_step']} |" for x in rows]
    # probes
    prow = []
    for name, r in R.items():
        f = r["dir"] / "eval" / "probe_walls.json"
        if f.exists():
            P = json.load(open(f))
            for s, v in P["sites"].items():
                prow.append(dict(model=name, site=s, abs_path=v["abs_walls_at_path_tokens"], abs_ps=v["abs_walls_at_PATH_START"], local4=v["local4_at_path_tokens"]))
    if prow:
        with open(HERE / "codes_probes.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(prow[0])); wr.writeheader(); wr.writerows(prow)
        md += ["", "| model | site | abs walls @ path tokens | abs walls @ PATH_START | local N/E/S/W @ path tokens |", "|---|---|---|---|---|"]
        md += [f"| {x['model']} | {x['site']} | {x['abs_path']:.3f} | {x['abs_ps']:.3f} | {x['local4']:.3f} |" for x in prow]
    drows, lrows = intact_rows(R)
    if drows:
        md += ["", "Intact prompt, standard set: solve rate on route-changed items (graft maze / prompt maze)", "",
               md_table(drows, ["model", "n", "same_maze", "route_changed_graft", "route_changed_prompt"])]
        with open(HERE / "codes_distance.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(drows[0])); wr.writeheader(); wr.writerows(drows)
    if lrows:
        md += ["", "Intact prompt, one-swap set (DFS family): graft carries the edited maze, prompt shows the old one", "",
               md_table(lrows, ["model", "n", "p_new", "top1_new", "gain_nats", "used_door", "crossed_wall", "solved_new", "prompt_only_p_new", "both_p_new"])]
        with open(HERE / "codes_local.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(lrows[0])); wr.writeheader(); wr.writerows(lrows)
    arows = ablate_rows(R)
    if arows:
        cols = ["model", "cond", "p_sp", "p_valid", "p_invalid", "solved", "reached", "clean", "valid_move_frac"]
        md += ["", "Inference-time ablations (true world, 2,000 mazes; prompt/generation split at the origin token)", "", md_table(arows, cols)]
        with open(HERE / "codes_ablate.csv", "w", newline="") as f:
            keys = sorted({k for x in arows for k in x}, key=lambda k: (k not in cols, cols.index(k) if k in cols else 0, k))
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(arows)
        # figure: one panel per condition metric (solved), models as grouped bars
        conds = [c for c in ["full", "graft_off", "off_gen", "off_prompt", "x0.5", "x2", "noise", "prompt_blank", "blank_off", "shuffled"] if any(x["cond"] == c for x in arows)]
        models = [m for m in R if any(x["model"] == m for x in arows)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, key, lab in ((axes[0], "solved", "solved (greedy rollout)"), (axes[1], "p_invalid", "p(invalid move) teacher-forced")):
            w = 0.8 / max(1, len(models))
            for i, m in enumerate(models):
                vals = [next((x[key] for x in arows if x["model"] == m and x["cond"] == c), np.nan) for c in conds]
                r = R[m]; ax.bar(np.arange(len(conds)) + (i - (len(models) - 1) / 2) * w, vals, w, color=PAL.get(r["code"], "#333"),
                                 alpha=1.0 if r["kind"] != "trainable" else 0.55, label=m, edgecolor="white", linewidth=0.5)
            ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels(conds, rotation=30, ha="right"); ax.set_title(lab, loc="left", fontsize=8)
        h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.22))
        fig.suptitle("6×6, seed 0, prompt intact — what happens when the graft is removed, scaled, replaced by noise, or is the only source", x=0.01, ha="left", fontsize=9)
        fig.tight_layout(); [fig.savefig(HERE / f"fig_codes_ablate.{e}", bbox_inches="tight") for e in ("pdf", "svg")]; plt.close(fig)
    open(HERE / "tables.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    figures(R, prow, ref, "intact")
    print("figures written")


def figures(R, prow, ref, subset):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True)
    for i, w in enumerate(("true", "phantom")):
        for j, (col, lab) in enumerate(((1, "p(shortest-path token)"), (2, "solved"))):
            ax = axes[i, j]
            for name, r in R.items():
                c = curves(r)
                if w in c:
                    ax.plot(c[w][:, 0], c[w][:, col], color=PAL.get(r["code"], "#333"), ls="-" if r["kind"] == "fixed" else "--", lw=1.6, label=name)
            for cond, colour, lab2 in (("base", BASE, "base, intact prompt (5 seeds, 1 ep)"), ("fixed", EDGE, "edge graft, intact prompt (5 seeds, 1 ep)")):
                if (cond, w) in ref:
                    ax.axhline(ref[(cond, w)][col - 1], color=colour, ls=":", lw=1, label=lab2)
            ax.set_ylim(-0.02, 1.02); ax.axvline(STEPS_PER_EPOCH, color="#bbb", lw=0.6)
            if i == 0: ax.set_title(lab, loc="left", fontsize=8)
            if i == 1: ax.set_xlabel("training step (7,812 = one pass over 2M mazes)")
            if j == 0: ax.set_ylabel(f"{w} world")
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("6×6, seed 0 — " + ("graft codes trained with the adjacency list blanked (graft = only source of the maze)" if subset == "blank"
                 else "graft codes with the prompt intact, injected from <ORIGIN_END> on (4 epochs)"), x=0.01, ha="left", fontsize=9)
    fig.tight_layout(); [fig.savefig(HERE / f"fig_codes_curves_{subset}.{e}", bbox_inches="tight") for e in ("pdf", "svg")]; plt.close(fig)
    if prow:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for j, (key, lab) in enumerate((("abs_path", "absolute walls, probed at path tokens"), ("abs_ps", "absolute walls, probed at <PATH_START>"), ("local4", "walls N/E/S/W of the current cell, at path tokens"))):
            ax = axes[j]
            for name, r in R.items():
                pts = [(0 if x["site"] == "L0in" else int(x["site"][1:]), x[key]) for x in prow if x["model"] == name]
                if not pts: continue
                if pts:
                    x, y = zip(*sorted(pts)); ax.plot(x, y, color=PAL.get(r["code"], "#333"), ls="-" if r["kind"] == "fixed" else "--", marker="o", ms=3, lw=1.5, label=name)
            ax.set_title(lab, loc="left", fontsize=8); ax.set_xlabel("layer (0 = block-0 input incl. graft; k = after block k-1)"); ax.set_ylim(0.45, 1.02)
            if j == 0: ax.set_ylabel("ridge-probe accuracy (held-out mazes)")
        h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle("Are the grafted features translated into wall features downstream? (" + subset + " prompt)", x=0.01, ha="left", fontsize=9)
        fig.tight_layout(); [fig.savefig(HERE / f"fig_codes_probes_{subset}.{e}", bbox_inches="tight") for e in ("pdf", "svg")]; plt.close(fig)


if __name__ == "__main__":
    main()
