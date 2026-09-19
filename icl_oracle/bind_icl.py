"""In-context latent-to-label binding with an injected residual-stream channel.

Task. A many-shot prompt of examples " item: label\n". Items are unique arbitrary words, labels are
two arbitrary words, and the label of each example is a deterministic function of a hidden bit z_n
that the TEXT does not contain. z_n is written into the residual stream at one position of the
example (the item token or the ':' readout token) as  s_n * beta * norm_ref * v,  s_n = +/-1.
The convention c (+v -> label A, or +v -> label B) is a property of the whole prompt and is
counterbalanced. A frozen model that comes to predict the label from the injection, with the sign
of the effect set by the convention, is using the channel in context; a fixed-sign effect is steering.

Arms (direction v):
  none     no injection (floor)                     token   z written as a marker WORD (ceiling)
  rand     random unit direction                    actpc   random unit direction in top-64 activation PCs
  native   diff-of-means direction of the marker (from the token arm, same layer/position)
  unembed  W_U[A]-W_U[B]  (a pure steering vector; the design must expose it as such)

One forward pass per prompt gives the whole shot curve (readout at every ':' is causal).
Stages: main (grid over arms x layers x beta x position), sig (onset / flip / xor signatures).
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("ICL_MODEL", "google/gemma-2-2b")
DEV = os.environ.get("ICL_DEV", "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
OUT = os.environ.get("ICL_OUT", os.path.join(HERE, "results_bind"))
os.makedirs(OUT, exist_ok=True)

N_EX = 128
K_PC = 64

ANIMALS = ("cat dog cow pig goat duck frog ant bee owl fox deer crab wolf bear lion horse fish bird "
           "mouse rat snake eagle shark whale seal camel zebra tiger sheep hen swan bat moth worm "
           "toad crow dove hawk mole elk bull calf lamb pony mule hare lynx puma yak goose turkey parrot pigeon "
           "sparrow robin falcon salmon trout tuna cod squid octopus lobster shrimp oyster clam snail spider beetle "
           "wasp fly mosquito butterfly cricket lizard turtle gecko cobra viper python gorilla monkey chimp baboon "
           "panda koala kangaroo otter beaver badger raccoon skunk squirrel hamster rabbit ferret weasel moose bison "
           "buffalo llama donkey hippo rhino giraffe leopard cheetah jaguar hyena jackal coyote dolphin walrus penguin "
           "pelican stork heron ostrich emu").split()
OBJECTS = ("apple car water sand math fire moon tree salt gold wind rain star book door milk iron song "
           "road snow king hand rock lamp ship cake glass river bread stone cloud paper chair table house "
           "green black white light night sun hill lake rose wine corn leaf ring bell coat shoe hat bag "
           "box cup key pen map flag ball drum coin desk wall roof gate bridge tower field farm boat "
           "train plane truck bus bike wheel engine motor queen judge nurse doctor pilot chef bank bone "
           "blood brain heart mouth tooth nose skin hair foot knee arm leg neck face eye ear lip seed "
           "root branch grass moss oak pine palm fern silk wool cotton leather steel copper silver tin "
           "lead brass clay chalk coal oil gas ice steam smoke dust mud ink soap candle mirror clock "
           "watch phone radio piano guitar violin flute knife fork spoon plate bowl pot pan oven sofa "
           "bed pillow blanket towel window floor stairs porch garden yard park street city town "
           "village island desert forest jungle ocean beach coast valley canyon cave planet comet "
           "rocket robot magnet laser crystal diamond pearl ruby jade amber").split()


class Lab:
    def __init__(self):
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEV).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d = self.model.config.hidden_size
        self.bos = self.tok.bos_token_id
        self.colon = self.single(":"); self.nl = self.single("\n")
        self.objects = [w for w in OBJECTS if self.single(" " + w) is not None]
        self.animals = [w for w in ANIMALS if self.single(" " + w) is not None]
        # large fallback pool of single-token lowercase words from the vocab (for long prompts)
        import re
        vocab = self.tok.get_vocab(); excl = set(OBJECTS) | set(ANIMALS)
        big = sorted({t.lstrip("\u2581\u0120") for t in vocab if re.fullmatch(r"[\u2581\u0120][a-z]{3,8}", t)} - excl)
        self.bigpool = [w for w in big if self.single(" " + w) is not None and len(self.tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
        self.hooks = []
        print(f"{MODEL} on {DEV}: {self.n_layers} layers, d={self.d}; pools obj={len(self.objects)} ani={len(self.animals)} big={len(self.bigpool)}; "
              f"load {time.time()-t0:.0f}s", flush=True)

    def single(self, s):
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    # ------------------------------------------------------------------ prompt construction
    def make_prompt(self, seed, c, token_arm=False, onset=0, flip=None, xor=False):
        """Returns dict(ids[T], item_pos[N], sep_pos[N], s[N], y[N] (1=label A), label_ids(2), c, n_tok)."""
        rng = np.random.default_rng(seed)
        if xor:
            ani = list(rng.choice(self.animals, N_EX // 2, replace=False))
            obj = list(rng.choice(self.objects, N_EX // 2 + 4, replace=False))
            items = ani + obj[:N_EX // 2]; cat = np.array([1] * (N_EX // 2) + [0] * (N_EX // 2))
            perm = rng.permutation(N_EX); items = [items[i] for i in perm]; cat = cat[perm]
            spare = obj[N_EX // 2:]
        else:
            pool = self.objects if len(self.objects) >= N_EX + 4 else self.bigpool
            words = list(rng.choice(pool, N_EX + 4, replace=False))
            items, spare = words[:N_EX], words[N_EX:]; cat = np.zeros(N_EX, int)
        labels, markers = spare[:2], spare[2:4]
        s = rng.choice([-1, 1], N_EX)
        s[:onset] = 0
        conv = np.full(N_EX, c)
        if flip is not None:
            conv[flip:] = -c
        z = np.where(s == 0, rng.choice([-1, 1], N_EX), s)      # onset: label is random before onset
        y = ((z * conv) > 0).astype(int)                          # 1 -> label A
        if xor:
            y = y ^ cat
        ids, item_pos, sep_pos, mark_pos = ([self.bos] if self.bos is not None else []), [], [], []
        lab_ids = [self.single(" " + labels[0]), self.single(" " + labels[1])]
        mk_ids = [self.single(" " + markers[0]), self.single(" " + markers[1])]
        for n in range(N_EX):
            if token_arm:
                mark_pos.append(len(ids)); ids.append(mk_ids[0] if z[n] > 0 else mk_ids[1])
            item_pos.append(len(ids)); ids.append(self.single(" " + items[n]))
            sep_pos.append(len(ids)); ids.append(self.colon)
            ids.append(lab_ids[0] if y[n] == 1 else lab_ids[1]); ids.append(self.nl)
        return dict(ids=np.array(ids), item_pos=np.array(item_pos), sep_pos=np.array(sep_pos), s=s, z=z, y=y,
                    conv=conv, cat=cat, label_ids=lab_ids, c=c, seed=seed, labels=labels, markers=markers)

    # ------------------------------------------------------------------ forward with injection
    def clear_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    @torch.inference_mode()
    def forward(self, prompts, deltas=None, inject_hs=None, hs_layers=()):
        """prompts: list of prompt dicts with equal length. deltas: [B,T,d] float32 numpy or None.
        Returns per-prompt readouts (logit diff, p2, mass) at sep positions and optional hidden states."""
        ids = torch.tensor(np.stack([p["ids"] for p in prompts]), device=DEV)
        if deltas is not None:
            dt = torch.tensor(deltas, dtype=torch.bfloat16, device=DEV)
            def fn(mod, inp, out):
                if isinstance(out, tuple):
                    out[0].add_(dt); return out
                out.add_(dt); return out
            self.hooks.append(self.model.model.layers[inject_hs - 1].register_forward_hook(fn))
        try:
            out = self.model.model(ids, output_hidden_states=bool(hs_layers))
        finally:
            self.clear_hooks()
        sp = torch.tensor(prompts[0]["sep_pos"], device=DEV)          # identical layout within a batch
        h_sep = out.last_hidden_state[:, sp]                            # [B,N,d] post final norm
        logits = self.model.lm_head(h_sep)
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        res = []
        for b, p in enumerate(prompts):
            lg = logits[b].float()                                    # [N, V]
            la, lb = lg[:, p["label_ids"][0]], lg[:, p["label_ids"][1]]
            pr = torch.softmax(lg, -1)
            pa, pb = pr[:, p["label_ids"][0]], pr[:, p["label_ids"][1]]
            top1 = lg.argmax(-1)
            r = dict(dlogit=(la - lb).cpu().numpy(), p2=(pa / (pa + pb)).cpu().numpy(), mass=(pa + pb).cpu().numpy(),
                     top_is_label=((top1 == p["label_ids"][0]) | (top1 == p["label_ids"][1])).float().cpu().numpy())
            if hs_layers:
                r["hs"] = {h: out.hidden_states[h][b].float().cpu().numpy() for h in hs_layers}
            res.append(r)
        return res

    def batched(self, prompts, deltas=None, inject_hs=None, hs_layers=(), bs=8):
        res = []
        for i in range(0, len(prompts), bs):
            d = None if deltas is None else deltas[i:i + bs]
            res += self.forward(prompts[i:i + bs], d, inject_hs, hs_layers)
        return res


# ---------------------------------------------------------------------- helpers
def pos_of(p, where):
    return p["item_pos"] if where == "item" else p["sep_pos"]


def make_delta(p, v, scale, where, sign=None):
    """[T,d] injection: sign_n * scale * v at the chosen position of every example."""
    T = len(p["ids"]); delta = np.zeros((T, len(v)), np.float32)
    s = p["s"] if sign is None else sign
    delta[pos_of(p, where)] = (s[:, None] * scale) * v[None]
    return delta


def summarize(r, p):
    """Strip to the arrays needed for analysis."""
    return dict(dlogit=r["dlogit"].astype(np.float32), p2=r["p2"].astype(np.float32), mass=r["mass"].astype(np.float32),
                top_is_label=r["top_is_label"].astype(np.float32), s=p["s"], z=p["z"], y=p["y"], conv=p["conv"],
                cat=p["cat"], c=p["c"], seed=p["seed"])


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, default=lambda a: a.tolist() if hasattr(a, "tolist") else a)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------- stage: main grid
def stage_main(lab, seeds, layers, betas, wheres, bs, arms=("rand", "actpc", "native", "unembed")):
    convs = [1, -1]
    prompts = {(sd, c): lab.make_prompt(sd, c) for sd in seeds for c in convs}
    tprompts = {(sd, c): lab.make_prompt(sd, c, token_arm=True) for sd in seeds for c in convs}
    keys = list(prompts)
    results = {"meta": dict(model=MODEL, seeds=seeds, layers=layers, betas=betas, wheres=wheres, n_ex=N_EX), "runs": {}}

    # floor + ceiling, with hidden states for reference norms / PCs / native directions
    t0 = time.time()
    rn = lab.batched([prompts[k] for k in keys], hs_layers=layers, bs=bs)
    rt = lab.batched([tprompts[k] for k in keys], hs_layers=layers, bs=bs)
    results["runs"]["none"] = [summarize(r, prompts[k]) for r, k in zip(rn, keys)]
    results["runs"]["token"] = [summarize(r, tprompts[k]) for r, k in zip(rt, keys)]
    print(f"floor/ceiling done {time.time()-t0:.0f}s", flush=True)

    # reference statistics per (layer, where)
    ref = {}
    for h in layers:
        for w in wheres:
            X = np.concatenate([r["hs"][h][pos_of(prompts[k], w)] for r, k in zip(rn, keys)])      # [S*N, d]
            mu = X.mean(0); Xc = X - mu
            norm_ref = float(np.median(np.linalg.norm(Xc, axis=1)))
            U, S, Vt = np.linalg.svd(Xc[::2].astype(np.float32), full_matrices=False)
            pcs = Vt[:K_PC]                                                                   # [K,d]
            Xt = np.concatenate([r["hs"][h][pos_of(tprompts[k], w)] for r, k in zip(rt, keys)])
            zt = np.concatenate([tprompts[k]["z"] for k in keys])
            native = unit(Xt[zt > 0].mean(0) - Xt[zt < 0].mean(0))
            # how visible is the native direction in the model's own activation PCs / what is its raw scale
            proj = float(np.linalg.norm(pcs @ native))
            nat_scale = float((Xt[zt > 0] @ native).mean() - (Xt[zt < 0] @ native).mean()) / 2
            ref[(h, w)] = dict(norm_ref=norm_ref, pcs=pcs, native=native, mu=mu, proj_native_in_pcs=proj,
                               native_halfgap=nat_scale, pc_var_frac=float((S[:K_PC] ** 2).sum() / (S ** 2).sum()))
            print(f"  ref h={h} {w}: norm_ref={norm_ref:.1f} native halfgap={nat_scale:.2f} "
                  f"|P native|={proj:.2f} top{K_PC} var={ref[(h,w)]['pc_var_frac']:.2f}", flush=True)
    results["meta"]["ref"] = {f"{h}_{w}": {k: v for k, v in d.items() if k in ("norm_ref", "proj_native_in_pcs", "native_halfgap", "pc_var_frac")}
                              for (h, w), d in ref.items()}

    # directions per seed
    def direction(arm, sd, h, w, p):
        rng = np.random.default_rng(10_000 + sd)
        if arm == "rand":
            return unit(rng.standard_normal(lab.d))
        if arm == "actpc":
            return unit(rng.standard_normal(K_PC) @ ref[(h, w)]["pcs"])
        if arm == "native":
            return ref[(h, w)]["native"]
        if arm == "unembed":
            W = lab.model.lm_head.weight[p["label_ids"]].detach().float().cpu().numpy()
            return unit(W[0] - W[1])
        raise ValueError(arm)

    arms = list(arms)
    n_cond = len(arms) * len(layers) * len(betas) * len(wheres)
    i = 0
    for arm in arms:
        for h in layers:
            for w in wheres:
                for b in betas:
                    t0 = time.time()
                    ps = [prompts[k] for k in keys]
                    ds = np.stack([make_delta(p, direction(arm, p["seed"], h, w, p), b * ref[(h, w)]["norm_ref"], w) for p in ps])
                    rr = lab.batched(ps, ds, h, bs=bs)
                    name = f"{arm}_h{h}_{w}_b{b}"
                    results["runs"][name] = [summarize(r, p) for r, p in zip(rr, ps)]
                    i += 1
                    acc = np.mean([np.mean(np.where(x["s"] * x["c"] > 0, x["p2"], 1 - x["p2"])[N_EX // 2:]) for x in results["runs"][name]])
                    print(f"[{i}/{n_cond}] {name}: late acc={acc:.3f} health={np.mean([x['top_is_label'][N_EX // 2:].mean() for x in results['runs'][name]]):.2f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    save("main.json", results)
    save("main.json", results)
    np.savez(os.path.join(OUT, "main_ref.npz"), **{f"{h}_{w}_{k}": v for (h, w), d in ref.items() for k, v in d.items() if isinstance(v, np.ndarray)})
    return results


# ---------------------------------------------------------------------- stage: signatures
def stage_sig(lab, seeds, conds, bs):
    """conds: list of (arm, h, where, beta). Runs onset / flip / xor for each, plus token+none references."""
    convs = [1, -1]
    ref = dict(np.load(os.path.join(OUT, "main_ref.npz")))
    for k, v in json.load(open(os.path.join(OUT, "main.json")))["meta"]["ref"].items():
        ref[f"{k}_norm_ref"] = v["norm_ref"]
    results = {"meta": dict(model=MODEL, seeds=seeds, conds=conds), "runs": {}}

    def direction(arm, sd, h, w, p):
        rng = np.random.default_rng(10_000 + sd)
        if arm == "rand": return unit(rng.standard_normal(lab.d))
        if arm == "actpc": return unit(rng.standard_normal(K_PC) @ ref[f"{h}_{w}_pcs"])
        if arm == "native": return ref[f"{h}_{w}_native"]
        if arm == "unembed":
            W = lab.model.lm_head.weight[p["label_ids"]].detach().float().cpu().numpy(); return unit(W[0] - W[1])

    tests = {"onset32": dict(onset=32), "flip64": dict(flip=64), "xor": dict(xor=True)}
    for tname, kw in tests.items():
        ps = [lab.make_prompt(sd, c, **kw) for sd in seeds for c in convs]
        tps = [lab.make_prompt(sd, c, token_arm=True, **kw) for sd in seeds for c in convs]
        results["runs"][f"{tname}_none"] = [summarize(r, p) for r, p in zip(lab.batched(ps, bs=bs), ps)]
        results["runs"][f"{tname}_token"] = [summarize(r, p) for r, p in zip(lab.batched(tps, bs=bs), tps)]
        for (arm, h, w, b) in conds:
            t0 = time.time()
            ds = np.stack([make_delta(p, direction(arm, p["seed"], h, w, p), b * float(ref[f"{h}_{w}_norm_ref"]), w) for p in ps])
            rr = lab.batched(ps, ds, h, bs=bs)
            name = f"{tname}_{arm}_h{h}_{w}_b{b}"
            results["runs"][name] = [summarize(r, p) for r, p in zip(rr, ps)]
            print(f"{name} done ({time.time()-t0:.0f}s)", flush=True)
            save("sig.json", results)
    save("sig.json", results)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--layers", default="")
    ap.add_argument("--betas", default="0.25,0.5,1,2")
    ap.add_argument("--wheres", default="item,sep")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--conds", default="")            # sig: arm:h:where:beta;...
    ap.add_argument("--n_ex", type=int, default=128)
    ap.add_argument("--arms", default="rand,actpc,native,unembed")
    a = ap.parse_args()
    N_EX = a.n_ex
    lab = Lab()
    seeds = list(range(a.seeds))
    L = lab.n_layers
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [max(1, L // 6), L // 3, L // 2, (2 * L) // 3]
    betas = [float(x) for x in a.betas.split(",")]
    wheres = a.wheres.split(",")
    if a.stage == "main":
        stage_main(lab, seeds, layers, betas, wheres, a.bs, a.arms.split(","))
    elif a.stage == "sig":
        conds = []
        for c in a.conds.split(";"):
            arm, h, w, b = c.split(":"); conds.append((arm, int(h), w, float(b)))
        stage_sig(lab, seeds, conds, a.bs)
