#!/usr/bin/env python3
"""Figure 1: illustrative counterfactual edit on a 6x6 maze.

Three panels, one maze, one prompt:
  A  base model, true world           (prompt only, no channel)
  B  injected model, true world       (channel carries the true adjacency)
  C  injected model, phantom world    (channel carries an edited adjacency:
                                       two phantom walls on the base rollout,
                                       two phantom doors where true walls are)

The maze is a random-DFS spanning tree (the training distribution). Rollouts
are BFS shortest paths in the world the model is assumed to navigate (the
true maze in A/B, the grafted maze in C) - an illustration, not a model run.

    python make_fig1.py search  --out /tmp/contact.png # browse candidates (scratch raster, not part of the figure set)
    python make_fig1.py render  --seed S --combo K     # final figure
    python make_fig1.py render --from-json fig1.json  # reproduce saved layout
"""

import itertools
import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch, Circle

N = 6

# ---- palette (dataviz reference instance, light mode) ----------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
BASE = "#2a7a57"        # base model: origin marker + rollout (user spec)
GRAFT = "#000c7a"       # grafted / injected model: origin marker + rollout
PATH = GRAFT
PHANTOM = "#eb6834"     # orange - grafted edits (phantom wall / door)
WALL_WIDTH = 2.0
PHANTOM_ALPHA = 0.75
WALL_SIZE = 0.089  # Shared wall thickness in grid coordinates.
ENDPOINT_GAP = 0.040
START_RADIUS = 0.14
TARGET_STROKE = 0.0623


# ---- maze primitives --------------------------------------------------------
def cells():
    return [(r, c) for r in range(N) for c in range(N)]


def nbrs(cell):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N:
            yield (rr, cc)


def all_adjacencies():
    out = set()
    for a in cells():
        for b in nbrs(a):
            out.add(frozenset((a, b)))
    return out


def gen_dfs(rng):
    """Random-DFS spanning tree over the grid (maze-dataset gen_dfs recipe)."""
    start = (int(rng.integers(N)), int(rng.integers(N)))
    visited = {start}
    stack = [start]
    edges = set()
    while stack:
        cur = stack[-1]
        cand = [b for b in nbrs(cur) if b not in visited]
        if cand:
            b = cand[int(rng.integers(len(cand)))]
            edges.add(frozenset((cur, b)))
            visited.add(b)
            stack.append(b)
        else:
            stack.pop()
    return edges


def bfs(edges, src):
    """dist + parent map + count of shortest paths (for uniqueness)."""
    adj = {}
    for e in edges:
        a, b = tuple(e)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    dist = {src: 0}
    par = {src: None}
    cnt = {src: 1}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj.get(u, [])):
            if v not in dist:
                dist[v] = dist[u] + 1
                par[v] = u
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    return dist, par, cnt


def shortest_path(edges, src, dst):
    dist, par, cnt = bfs(edges, src)
    if dst not in dist:
        return None, 0
    path = [dst]
    while path[-1] != src:
        path.append(par[path[-1]])
    return path[::-1], cnt[dst]


def path_edges(path):
    return [frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)]


def turns(path):
    t = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        t += d1 != d2
    return t


# ---- candidate search -------------------------------------------------------
def candidates_for_seed(seed, min_base=10, max_base=16, max_per_seed=6):
    rng = np.random.default_rng(seed)
    edges = gen_dfs(rng)
    walls = all_adjacencies() - edges
    # origin/target: longest tree path, ties -> most turns
    # origin/target: a mid-length, winding tree path (not the diameter)
    pairs = []
    for o in cells():
        dist, par, _ = bfs(edges, o)
        for t, d in dist.items():
            if not (min_base <= d <= max_base) or o >= t:
                continue
            p, _ = shortest_path(edges, o, t)
            pairs.append((turns(p) + 0.15 * d, o, t, p))
    pairs.sort(key=lambda x: -x[0])
    out = []
    for _, o, t, base in pairs[:3]:
        out += _combos_for_pair(seed, edges, walls, o, t, base)
    out.sort(key=lambda c: -c["score"])
    return out[:max_per_seed]


def _mid(e):
    x0, y0, x1, y1 = seg(e)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _combos_for_pair(seed, edges, walls, o, t, base):
    bedges = path_edges(base)
    out = []
    for (i1, w1), (i2, w2) in itertools.combinations(list(enumerate(bedges)), 2):
        if i1 < 1 or i2 - i1 < 3 or i2 >= len(bedges) - 1:
            continue
        core = edges - {w1, w2}
        for d1, d2 in itertools.combinations(sorted(walls, key=lambda e: sorted(e)), 2):
            g = core | {d1, d2}
            p, ncnt = shortest_path(g, o, t)
            if p is None or ncnt != 1:
                continue
            pe = set(path_edges(p))
            if d1 not in pe or d2 not in pe:
                continue
            # each door necessary: dropping it lengthens / disconnects
            ok = True
            for d in (d1, d2):
                q, _ = shortest_path(g - {d}, o, t)
                if q is not None and len(q) <= len(p):
                    ok = False
            if not ok:
                continue
            # each phantom wall load-bearing: un-asserting it re-routes through it
            for w in (w1, w2):
                q, _ = shortest_path(g | {w}, o, t)
                if q is None or len(q) >= len(p) or w not in set(path_edges(q)):
                    ok = False
            if not ok:
                continue
            # shares an opening prefix with the base rollout (visible deflection)
            pref = 0
            while pref < min(len(p), len(base)) and p[pref] == base[pref]:
                pref += 1
            if pref < 2:
                continue
            L0, L1 = len(base) - 1, len(p) - 1
            if L1 < 8 or L1 > 18:
                continue
            # edits spread out, not clustered in one corner
            mids = [_mid(e) for e in (w1, w2, d1, d2)]
            dmin = min(abs(a[0] - b[0]) + abs(a[1] - b[1])
                       for a, b in itertools.combinations(mids, 2))
            if dmin < 1.5:
                continue
            shared = len(pe & set(bedges))
            score = (min(pref, 4) * 1.5 + turns(p) + turns(base)
                     - 0.6 * abs(L1 - L0) + 0.8 * dmin
                     - 0.4 * abs(shared - L0 // 2))
            out.append(dict(seed=seed, score=score, origin=o, target=t,
                            edges=edges, base=base, phantom=p,
                            walls_added=[w1, w2], doors=[d1, d2]))
    return out


# ---- rendering --------------------------------------------------------------
def seg(e):
    """Line segment (x0,y0,x1,y1) for the wall between the two cells of e.
    Row r is drawn at y=-r (row 1 at the top), col c at x=c."""
    (r1, c1), (r2, c2) = sorted(e)
    if r1 == r2:                      # horizontal neighbours -> vertical wall
        x = c1 + 0.5
        return (x, -(r1 - 0.5), x, -(r1 + 0.5))
    y = -(r1 + 0.5)                   # vertical neighbours -> horizontal wall
    return (c1 - 0.5, y, c1 + 0.5, y)


def draw_frame(ax, labels=True):
    ax.set_facecolor(SURFACE)
    for k in range(N + 1):
        ax.plot([-0.5, N - 0.5], [-(k - 0.5)] * 2, color=GRID, lw=0.6, zorder=1, snap=False)
        ax.plot([k - 0.5] * 2, [0.5, -(N - 0.5)], color=GRID, lw=0.6, zorder=1, snap=False)
    h = WALL_SIZE / 2
    _fill_rectangles(ax, [(-0.5-h, -N+0.5-h, WALL_SIZE, N+2*h),
                          (N-0.5-h, -N+0.5-h, WALL_SIZE, N+2*h),
                          (-0.5-h, 0.5-h, N+2*h, WALL_SIZE),
                          (-0.5-h, -N+0.5-h, N+2*h, WALL_SIZE)], INK, 4)
    if labels:  # (row, col) coordinates: rows down the left, cols along the bottom
        for k in range(N):
            ax.text(k, -(N - 0.5) - 0.2, str(k + 1), ha="center", va="top",
                    fontsize=7, color=MUTED)
            ax.text(-0.5 - 0.22, -k, str(k + 1), ha="right", va="center",
                    fontsize=7, color=MUTED)
    ax.set_xlim(-1.05, N - 0.35)
    ax.set_ylim(-(N + 0.25), 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def _fill_rectangles(ax, rectangles, color, z, alpha=1.0):
    """Fill a compound shape once; no stroked edges or overlapping alpha."""
    paths = []
    for x, y, width, height in rectangles:
        paths.append(PlotPath([(x, y), (x+width, y), (x+width, y+height),
                               (x, y+height), (x, y)],
                              [1, 2, 2, 2, 79]))
    if paths:
        ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths),
                               facecolor=color, edgecolor="none", linewidth=0,
                               alpha=alpha, zorder=z, snap=False))


def draw_walls(ax, wall_set, color=INK, lw=WALL_WIDTH, ls="-", z=3, alpha=1.0):
    junctions = {}
    rectangles = []
    h = WALL_SIZE * lw / WALL_WIDTH / 2
    for e in wall_set:
        x0, y0, x1, y1 = seg(e)
        orientation = "vertical" if x0 == x1 else "horizontal"
        for point in ((x0, y0), (x1, y1)):
            junctions.setdefault(point, set()).add(orientation)
        rectangles.append((min(x0, x1) - (h if x0 == x1 else 0),
                           min(y0, y1) - (h if y0 == y1 else 0),
                           abs(x1-x0) or 2*h, abs(y1-y0) or 2*h))
    # Fill the tiny outside corner only where perpendicular true walls meet.
    # Exposed ends remain flush, including those beside phantom edits.
    if ls == "-":
        for (x, y), orientations in junctions.items():
            if len(orientations) == 2:
                rectangles.append((x-h, y-h, 2*h, 2*h))
    _fill_rectangles(ax, rectangles, color, z, alpha)


def draw_phantom_edits(ax, doors, walls, true_walls):
    """Keep edits clear of junctions and leave room for door crossings."""
    # Explicit, symmetric dashes make both sides of every opening identical.
    # The central gap keeps the orange marks separate from the navy rollout.
    h = WALL_SIZE / 2
    door_spans = ((h, 0.20), (0.28, 0.42),
                  (0.58, 0.72), (0.80, 1-h))
    true_segments = [seg(e) for e in true_walls]
    rectangles = []
    for wall_set, spans in (
        (doors, door_spans),
        (walls, ((0.0, 1.0),)),
    ):
        for e in wall_set:
            x0, y0, x1, y1 = seg(e)
            # Meet the visible edge of each perpendicular black wall exactly.
            def clearance(x, y):
                frame = x in (-0.5, N-0.5) or y in (0.5, -N+0.5)
                perpendicular = any(
                    ((x0 == x1) != (a == c)) and
                    min(a, c) <= x <= max(a, c) and min(b, d) <= y <= max(b, d)
                    for a, b, c, d in true_segments)
                return h if frame or perpendicular else 0.0
            for start, stop in spans:
                start = max(start, clearance(x0, y0))
                stop = min(stop, 1-clearance(x1, y1))
                a, b = x0 + start*(x1-x0), y0 + start*(y1-y0)
                c, d = x0 + stop*(x1-x0), y0 + stop*(y1-y0)
                rectangles.append((min(a, c)-(h if a == c else 0),
                                   min(b, d)-(h if b == d else 0),
                                   abs(c-a) or 2*h, abs(d-b) or 2*h))
    # A neutral backing prevents grid lines and reference dots showing through
    # as dark stripes, while retaining the requested translucent orange tone.
    _fill_rectangles(ax, rectangles, SURFACE, 4.7)
    _fill_rectangles(ax, rectangles, PHANTOM, 4.8, PHANTOM_ALPHA)


def _fill_polygons(ax, polygons, color, z, alpha=1.0):
    # Consistent winding makes overlapping pieces a union, never a cutout.
    polygons = [vertices if np.sum(vertices[:, 0]*np.roll(vertices[:, 1], -1)
                                  - vertices[:, 1]*np.roll(vertices[:, 0], -1)) > 0
                else vertices[::-1] for vertices in polygons]
    paths = [PlotPath(np.vstack((vertices, vertices[0])),
                      [PlotPath.MOVETO] + [PlotPath.LINETO]*(len(vertices)-1)
                      + [PlotPath.CLOSEPOLY]) for vertices in polygons]
    ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths),
                           facecolor=color, edgecolor="none", linewidth=0,
                           alpha=alpha, zorder=z, snap=False))


def draw_path(ax, path, color=PATH, lw=1.8, z=5, alpha=1.0):
    points = np.array([(c, -r) for r, c in path], dtype=float)
    if len(points) < 2:
        return
    first_direction = points[1] - points[0]
    last_direction = points[-1] - points[-2]
    half_width = WALL_SIZE * lw / WALL_WIDTH / 2
    socket_radius = START_RADIUS + ENDPOINT_GAP
    # The X's incoming notch and the arrow sides are all exactly 45 degrees.
    # Match their perpendicular gap to the radial circle/shaft gap.
    tip_distance = TARGET_STROKE / np.sqrt(2) + np.sqrt(2)*ENDPOINT_GAP
    head_half_width = 3.0*half_width
    base_distance = tip_distance + head_half_width
    stroke = points.copy()
    stroke[0] += socket_radius * first_direction
    stroke[-1] -= base_distance * last_direction
    polygons = []
    for a, b in zip(stroke, stroke[1:]):
        direction = (b-a) / np.linalg.norm(b-a)
        side = np.array([-direction[1], direction[0]]) * half_width
        polygons.append(np.array([a+side, b+side, b-side, a-side]))
    # Concave shaft termination follows a concentric circle: no extra tail arc.
    side = np.array([-first_direction[1], first_direction[0]])
    transverse = np.linspace(-half_width, half_width, 41)
    boundary = (points[0] + np.sqrt(socket_radius**2-transverse**2)[:, None]
                * first_direction + transverse[:, None]*side)
    back = points[0] + (socket_radius+0.01)*first_direction
    polygons.append(np.vstack((back-half_width*side, boundary,
                               back+half_width*side)))
    side = np.array([-last_direction[1], last_direction[0]])
    end = points[-1]
    polygons.append(np.array([end-tip_distance*last_direction,
                               end-base_distance*last_direction+head_half_width*side,
                               end-base_distance*last_direction-head_half_width*side]))
    _fill_polygons(ax, polygons, color, z, alpha)
    for point in points[1:-1]:
        ax.add_patch(Circle(point, radius=0.089, facecolor=color,
                            edgecolor="none", alpha=alpha, zorder=z+1))


def draw_endpoints(ax, origin, target, color=PATH):
    ax.add_patch(Circle((origin[1], -origin[0]), radius=START_RADIUS,
                        facecolor=color, edgecolor="none", zorder=7))
    center = np.array([target[1], -target[0]], dtype=float)
    arms = []
    for direction in (np.array([1., 1.]), np.array([1., -1.])):
        normal = np.array([-direction[1], direction[0]]) / np.sqrt(2)
        a, b = center-0.165*direction, center+0.165*direction
        offset = TARGET_STROKE/2 * normal
        arms.append(np.array([a+offset, b+offset, b-offset, a-offset]))
    _fill_polygons(ax, arms, color, 7)


def render(cand, out_stem, title_size=8.5):
    edges = cand["edges"]
    walls = all_adjacencies() - edges
    wa = set(cand["walls_added"])
    doors = set(cand["doors"])
    o, t = cand["origin"], cand["target"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    # High-resolution raster export avoids uneven antialiasing of tiny marks.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), dpi=600)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.12,
                        wspace=0.08)

    heads = [
        ("Baseline Model", ""),
        ("Grafted Features: Original Maze", ""),
        ("Grafted Features: Edited Maze", ""),
    ]
    for ax, (h1, h2) in zip(axes, heads):
        draw_frame(ax)
        ax.text(0.5, 1.0, h1, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=title_size, color=INK, weight="bold")
        ax.text(0.5, 0.965, h2, transform=ax.transAxes, ha="center",
                va="top", fontsize=title_size - 0.5, color=INK2)

    # A: base model, true world (green). B: grafted model, true world (navy)
    for ax, col in zip(axes[:2], (BASE, GRAFT)):
        draw_walls(ax, walls)
        draw_path(ax, cand["base"], color=col)
        draw_endpoints(ax, o, t, color=col)

    # C: grafted world = true walls - doors + phantom walls
    ax = axes[2]
    draw_walls(ax, walls - doors)
    draw_phantom_edits(ax, doors, wa, walls - doors)
    draw_path(ax, cand["phantom"], color=GRAFT)
    draw_endpoints(ax, o, t, color=GRAFT)

    for ax, path in zip(axes, (cand["base"], cand["base"], cand["phantom"])):
        ax.text((N-1)/2, -N-0.40, f"{len(path)-1} steps", ha="center",
                va="center", fontsize=7, color=INK2)

    handles = [
        Line2D([], [], color=PHANTOM, lw=WALL_WIDTH, alpha=PHANTOM_ALPHA,
               solid_capstyle="butt", label="phantom wall"),
        Line2D([], [], color=PHANTOM, lw=WALL_WIDTH, alpha=PHANTOM_ALPHA,
               ls=(0, (2.2, 1.6)), dash_capstyle="butt",
               label="phantom door"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=7, handlelength=3.0,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))

    for ext in ("pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", facecolor=SURFACE)
    plt.close(fig)


def contact_sheet(cands, out):
    n = len(cands)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows),
                             dpi=110)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, cand in zip(np.ravel(axes), cands):
        walls = all_adjacencies() - cand["edges"]
        doors = set(cand["doors"])
        wa = set(cand["walls_added"])
        draw_frame(ax)
        draw_walls(ax, walls - doors)
        draw_phantom_edits(ax, doors, wa, walls - doors)
        draw_path(ax, cand["phantom"])
        draw_endpoints(ax, cand["origin"], cand["target"])
        ax.set_title(f"seed {cand['seed']} combo {cand['combo']}  "
                     f"L0={len(cand['base'])-1} L1={len(cand['phantom'])-1}  "
                     f"s={cand['score']:.1f}", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def dump(cand, path):
    j = dict(
        grid=N, seed=cand["seed"], origin=list(cand["origin"]),
        target=list(cand["target"]),
        true_edges=sorted(sorted(map(list, e)) for e in cand["edges"]),
        base_rollout=list(map(list, cand["base"])),
        phantom_walls=[sorted(map(list, e)) for e in cand["walls_added"]],
        phantom_doors=[sorted(map(list, e)) for e in cand["doors"]],
        phantom_rollout=list(map(list, cand["phantom"])),
        note="0-indexed (row, col); row 0 is the top row. Rollouts are BFS "
             "shortest paths in the respective world (illustrative).",
    )
    Path(path).write_text(json.dumps(j, indent=1))


def load_candidate(path):
    """Reproduce an explicitly selected maze without rerunning candidate search."""
    j = json.loads(Path(path).read_text())
    if j["grid"] != N:
        raise ValueError(f"Expected a {N}x{N} maze")
    return dict(seed=j["seed"], origin=tuple(j["origin"]), target=tuple(j["target"]),
                edges={frozenset(map(tuple, e)) for e in j["true_edges"]},
                walls_added=[frozenset(map(tuple, e)) for e in j["phantom_walls"]],
                doors=[frozenset(map(tuple, e)) for e in j["phantom_doors"]],
                base=list(map(tuple, j["base_rollout"])),
                phantom=list(map(tuple, j["phantom_rollout"])))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "render"])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--from-json", type=Path,
                    help="Render an existing layout JSON (relative to this script)")
    ap.add_argument("--combo", type=int, default=0)
    ap.add_argument("--out", default="fig1")
    a = ap.parse_args()
    here = Path(__file__).parent
    if a.mode == "search":
        cands = []
        for s in range(a.seed0, a.seed0 + a.seeds):
            cs = candidates_for_seed(s)
            for k, c in enumerate(cs):
                c["combo"] = k
            cands += cs[:2]
        cands.sort(key=lambda c: -c["score"])
        cands = cands[:a.top]
        contact_sheet(cands, here / a.out)
        for c in cands:
            print(f"seed {c['seed']} combo {c['combo']} score {c['score']:.1f} "
                  f"L0={len(c['base'])-1} L1={len(c['phantom'])-1}")
    else:
        if a.from_json:
            cand = load_candidate(here / a.from_json)
        else:
            if a.seed is None:
                ap.error("render requires --seed or --from-json")
            cs = candidates_for_seed(a.seed)
            cand = cs[a.combo]
            cand["combo"] = a.combo
        figs = here / "figures"
        figs.mkdir(exist_ok=True)
        stem = figs / a.out
        render(cand, stem)
        dump(cand, here / f"{a.out}.json")          # layout JSON stays next to the script
        print(f"wrote figures/{a.out}.{{pdf,svg}} + {a.out}.json  "
              f"L0={len(cand['base'])-1} L1={len(cand['phantom'])-1}")


if __name__ == "__main__":
    main()
