"""Build 06_tripolar_fold.ipynb: visualize xgcm's bipolar north fold directly in
logical grid-index space (no map projection). For three real tripolar models we
show that the fold fills the northern halo with the interior *reflected about the
poles* (sign-flipped for velocities), so fields stay continuous across the seam
while the naive `extend` boundary just smears the edge."""
import json, os
EX = "/Users/hfdrake/code/xgcm/docs/xgcm-examples"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


cells = [
    md(r"""
# The bipolar north fold (on tripolar grids): do `diff` and `interp` work across the seam?

Global ocean grids like MOM6, NEMO, and Oceananigans are **tripolar** — the
South Pole plus two poles displaced over Arctic land — and their northern edge
folds onto itself along the **bipolar seam** joining the two northern poles. To
evaluate `interp`/`diff`/`derivative` across that edge, xgcm pads a northern
**halo** by reflecting the interior about the nearest pole (sign-flipping vector
components). If that reconstruction is correct, a physically smooth field stays
**smooth across the seam**.

Rather than map the Arctic (where a polar projection distorts the seam, land
masks it with a blank line, and model noise muddies the picture), we look at the
fold **directly in logical grid-index space** — x-index across, y-index up. The
seam is then just a horizontal line, and the fold's action is plain to see:

* the **halo is the interior reflected about the poles** — so any structure at
  the seam is the model's own field mirrored, *not* a fold artifact;
* the naive `extend` boundary instead **smears the edge value** upward;
* walking a column **across** the seam, the fold **continues** the real field
  while naive **flatlines**.

We show this for **three models** (real surface velocities `u`,`v`): MOM6
(GFDL-CM4) and NEMO (IPSL-CM6A-LR) use a `"corner"` fold pivot, Oceananigans a
`"u"` pivot.

> **Dependencies** — the MOM6/NEMO sections read CMIP6 from the Pangeo cloud
> (`pip install zarr gcsfs`).
"""),
    code(r"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from xgcm import Grid
from xgcm.padding import pad

so = {"storage_options": {"token": "anon"}}
"""),
    md(r"""
## Helpers

`package` puts a model's `u`,`v` on a common staggered index grid and masks land
(cells whose velocity is missing, or zero where zeros dominate — e.g. the
Oceananigans immersed boundary), so land never leaks through `interp`/`diff`.
The diagnostics each cross the fold through a different operation and grid
position: `speed_centre`/`speed_corner` test `interp` (to the cell centre and
corner), and `divergence` tests `diff` (its `∂v/∂y` term crosses the seam).
"""),
    code(r"""
def package(uo, vo, lon, lat, fold, label):
    '''Bundle surface velocities on a common staggered index grid, masking land.'''
    a = lambda x: np.asarray(getattr(x, "values", x), dtype=float)
    uo, vo, lon, lat = a(uo), a(vo), a(lon), a(lat)
    # land = missing, or (where zeros dominate, e.g. immersed boundaries) zero
    def mask_land(z):
        zz = np.where(np.isfinite(z), z, np.nan)
        if np.mean(z == 0) > 0.05:
            zz = np.where(z == 0, np.nan, zz)
        return zz
    uo, vo = mask_land(uo), mask_land(vo)
    ny, nx = uo.shape
    coords = dict(x_c=np.arange(nx), x_f=np.arange(nx), y_c=np.arange(ny), y_f=np.arange(ny))
    u = xr.DataArray(uo, dims=["y_c", "x_f"]).assign_coords(x_f=coords["x_f"], y_c=coords["y_c"])
    v = xr.DataArray(vo, dims=["y_f", "x_c"]).assign_coords(x_c=coords["x_c"], y_f=coords["y_f"])
    return dict(coords=coords, u=u, v=v, lon=lon, lat=lat, fold=fold, label=label)


def _grid(coords, edge, ybc):
    return Grid(xr.Dataset(coords=coords),
                coords={"X": {"center": "x_c", edge: "x_f"},
                        "Y": {"center": "y_c", edge: "y_f"}},
                boundary={"X": "periodic", "Y": ybc}, autoparse_metadata=False)


def speed_centre(m, fold=True):
    '''sqrt(u^2+v^2) at tracer points; the v->centre interp crosses the fold.'''
    g = _grid(m["coords"], "left", {"fold": m["fold"]} if fold else "extend")
    uc = g.interp(m["u"], "X")
    vc = (g.interp({"Y": m["v"]}, "Y", other_component={"X": m["u"]}, boundary="extend")
          if fold else g.interp(m["v"], "Y", boundary="extend"))
    return np.hypot(uc, vc)


def speed_corner(m, fold=True):
    '''sqrt(u^2+v^2) at cell corners; the u->corner interp crosses the fold.'''
    g = _grid(m["coords"], "right", {"fold": m["fold"]} if fold else "extend")
    vc = g.interp(m["v"], "X")
    uc = (g.interp({"X": m["u"]}, "Y", other_component={"Y": m["v"]}, boundary="extend")
          if fold else g.interp(m["u"], "Y", boundary="extend"))
    return np.hypot(uc, vc)


def divergence(m, fold=True):
    '''Horizontal divergence du/dx + dv/dy (per grid cell) at the cell centre;
    the dv/dy diff crosses the fold, so this exercises `diff` (and the vector
    fold of v) across the seam. Unlike vorticity, divergence is a TRUE SCALAR —
    invariant under the 180° fold — so it is continuous across the seam and its
    halo is simply the mirrored interior, with no sign subtlety.'''
    g = _grid(m["coords"], "left", {"fold": m["fold"]} if fold else "extend")
    dudx = g.diff(m["u"], "X", boundary="fill")
    dvdy = (g.diff({"Y": m["v"]}, "Y", other_component={"X": m["u"]}, boundary="fill")
            if fold else g.diff(m["v"], "Y", boundary="extend"))
    return dudx + dvdy
"""),
    md(r"""
## Seam-space plotting

These draw a zoom of the top rows in logical index space: the interior just
below the seam line, and `K` halo rows above it, reconstructed by `pad(...,
boundary_width={"Y": (0, K)})`. `attach_windows` picks **one** window of columns
over **open water** per model (the displaced poles themselves sit over Arctic
land, so the seam is ocean *between* them) and **all three figures below reuse
that same window**, so the panels are directly comparable. Cells are drawn with
nearest-neighbour shading, so every individual grid cell is a crisp, readable
block rather than a smoothed image.
"""),
    code(r"""
def _pad_scalar(S, m, K, mode):
    '''K halo rows above a centre scalar field: fold (mirror) or extend (smear).'''
    g = _grid(m["coords"], "right", {"fold": m["fold"]} if mode == "fold" else "extend")
    return np.asarray(pad(S, g, boundary_width={"Y": (0, K)}).values)


def _pad_v(m, K, mode, vector):
    '''K halo rows above the v component. vector=True folds it as a vector (the
    180° pivot flips its sign); vector=False folds it as a plain scalar. v lives
    at (y_f, x_c); for the vector case u is the other (X) component.'''
    g = _grid(m["coords"], "left", {"fold": m["fold"]} if mode == "fold" else "extend")
    if vector:
        out = pad({"Y": m["v"]}, g, boundary_width={"Y": (0, K)}, other_component={"X": m["u"]})
    else:
        out = pad(m["v"], g, boundary_width={"Y": (0, K)})
    return np.asarray(out.values)


def _pad_u(m, K, mode, vector):
    '''K halo rows above the u component (zonal). vector=True folds it as a
    vector (the 180° pivot flips its sign); vector=False as a plain scalar. u
    lives at (y_c, x_f); for the vector case v is the other (Y) component.'''
    g = _grid(m["coords"], "left", {"fold": m["fold"]} if mode == "fold" else "extend")
    if vector:
        out = pad({"X": m["u"]}, g, boundary_width={"Y": (0, K)}, other_component={"Y": m["v"]})
    else:
        out = pad(m["u"], g, boundary_width={"Y": (0, K)})
    return np.asarray(out.values)


def _ocean_window(rows, nx, W):
    '''Start column of the length-W window (wrapping in x) with the most ocean
    (finite cells) in the given rows. The poles sit over land, so we centre on
    open water rather than on a pole. Returns (start, column indices).'''
    fin = np.isfinite(rows).sum(axis=0).astype(int)
    best, score = 0, -1
    for s in range(nx):
        sc = int(fin[(np.arange(s, s + W)) % nx].sum())
        if sc > score:
            score, best = sc, s
    return best, (np.arange(best, best + W)) % nx


LAND = "0.7"   # grey for masked (land) cells, distinct from every colormap


def _imshow(ax, arr_win, ylo, **kw):
    '''Draw an already-windowed strip with nearest-neighbour shading, so every
    grid cell is a crisp, individually visible block (no interpolation blur).
    Masked (land) cells show as grey, distinct from the colormaps.'''
    nrow, ncol = arr_win.shape
    ax.set_facecolor(LAND)
    im = ax.imshow(arr_win, origin="lower", aspect="auto", interpolation="nearest",
                   extent=[-0.5, ncol - 0.5, ylo - 0.5, ylo + nrow - 0.5], **kw)
    ax.set_xticks([0, ncol // 2, ncol - 1])
    return im


def fill_seam_vrow(m):
    '''Reconstruct the redundant top row of the meridional velocity ``v``.

    On a tripolar F-fold the top ``v`` faces lie on the seam and come in fold
    pairs ``v(i) = -v(mirror(i))``; the CMIP6 NEMO/MOM6 products keep one half of
    that row and mask the other (the redundant duplicate). xgcm's fold fills
    *halos* from the interior, not interior gaps, so we rebuild this seam row
    here from its fold partner. Without it, interpolating ``v`` to the cell
    centre (e.g. for surface speed) would leave a blank band at the seam. Only
    masked cells whose fold partner is present are filled (a no-op for fields
    that are already complete, e.g. the Oceananigans simulation output).'''
    v = np.asarray(m["v"].values).copy()
    nx = v.shape[1]
    mirror = (-np.arange(nx) - 1) % nx          # seam reflection (seam axis = edge)
    top = v[-1]
    fill = ~np.isfinite(top) & np.isfinite(top[mirror])
    top[fill] = -top[mirror][fill]              # v flips sign across the fold
    m["v"] = m["v"].copy(data=v)


def attach_windows(models, K=6, W=28):
    '''Pick one open-water column window per model (from the fold-padded centre
    speed) and store it on the model, so every seam figure below shows the SAME
    region for that model and the panels are directly comparable.'''
    for m in models:
        S = speed_centre(m, True)
        ny, nx = S.sizes["y_c"], S.sizes["x_c"]
        Sf = _pad_scalar(S, m, K, "fold")[ny - K:ny + K]
        start, cols = _ocean_window(Sf, nx, W)
        m["win"] = (start, cols, W)


def seam_strip(models, K=6, W=28):
    '''Surface speed near the seam, zoomed to W cells around a pole so each grid
    cell is individually visible: naive halo / fold halo / difference. The fold
    halo mirrors the interior about the pole; naive 'extend' smears the edge; the
    interior (below the line) is untouched.'''
    rlab = ["naive halo\n(extend)", "fold halo\n(mirror)", "naive − fold"]
    fig, axes = plt.subplots(3, len(models), figsize=(4.6 * len(models), 9.4))
    for c, m in enumerate(models):
        S = speed_centre(m, True)
        ny, nx = S.sizes["y_c"], S.sizes["x_c"]
        Sf = _pad_scalar(S, m, K, "fold")[ny - K:ny + K]
        Se = _pad_scalar(S, m, K, "extend")[ny - K:ny + K]
        start, cols, W = m["win"]
        seam = S.values[ny - K:ny]
        vmax = np.nanpercentile(seam[np.isfinite(seam)], 95) if np.isfinite(seam).any() else 1.0
        vmax = vmax or 1.0
        dd = (Se - Sf) / vmax                                     # difference in units of vmax
        seq = plt.get_cmap("viridis").copy(); seq.set_bad(LAND)
        div = plt.get_cmap("RdBu_r").copy(); div.set_bad(LAND)
        for r, arr, cmap, kw in [
            (0, Se / vmax, seq, dict(vmin=0, vmax=1)),
            (1, Sf / vmax, seq, dict(vmin=0, vmax=1)),
            (2, dd, div, dict(vmin=-1, vmax=1)),
        ]:
            ax = axes[r, c]
            _imshow(ax, arr[:, cols], ny - K, cmap=cmap, **kw)
            ax.axhline(ny - 0.5, color="k", lw=1.6)              # the fold seam
            if r == 0:
                ax.set_title(f"{m['label']}\n(cols {start}–{start + W - 1})", fontsize=9)
            if c == 0:
                ax.set_ylabel(rlab[r], fontsize=9)
            if r == 2:
                ax.set_xlabel("X index (windowed)")
    for r, lab in [(0, "speed / max"), (1, "speed / max"), (2, "(naive−fold) / max")]:
        fig.colorbar(axes[r, -1].images[0],
                     ax=list(axes[r, :]), shrink=0.7, pad=0.02, label=lab)
    fig.suptitle("Surface speed near the seam (per-model scale, zoomed to open water so each cell "
                 "is visible):\nthe fold halo is the real cross-seam field (structured); naive "
                 "'extend' smears the edge straight up; interior below the line is untouched.",
                 fontsize=12, y=0.98)
    plt.show()


def component_strip(models, K=6, W=28):
    '''Both velocity components near the seam, zoomed to open water. For each of
    v (meridional) and u (zonal) the upper panel folds the component as a plain
    scalar and the lower as a vector. In the halo (above the line) the vector
    fold is the scalar fold with its sign flipped — the colours invert — for
    BOTH components: the 180° pivot reverses velocities, while a true scalar
    would not. (Interior, below the line, is identical for the two folds.)'''
    panels = [("v", False, "v folded as\nscalar"), ("v", True, "v folded as\nvector"),
              ("u", False, "u folded as\nscalar"), ("u", True, "u folded as\nvector")]
    pad_fn = {"v": _pad_v, "u": _pad_u}
    fig, axes = plt.subplots(len(panels), len(models),
                             figsize=(4.6 * len(models), 2.7 * len(panels)))
    for c, m in enumerate(models):
        ny, nx = m["coords"]["y_c"].size, m["coords"]["x_c"].size
        start, cols, W = m["win"]
        div = plt.get_cmap("RdBu_r").copy(); div.set_bad(LAND)
        scale = {}                          # one symmetric scale per component
        for comp in ("v", "u"):
            a = pad_fn[comp](m, K, "fold", True)[ny - K:ny + K]
            s = np.nanpercentile(np.abs(a[np.isfinite(a)]), 95) if np.isfinite(a).any() else 1.0
            scale[comp] = s or 1.0
        for r, (comp, vec, lab) in enumerate(panels):
            arr = pad_fn[comp](m, K, "fold", vec)[ny - K:ny + K]
            ax = axes[r, c]
            _imshow(ax, arr[:, cols] / scale[comp], ny - K, cmap=div, vmin=-1, vmax=1)
            ax.axhline(ny - 0.5, color="k", lw=1.6)              # the fold seam
            if r == 0:
                ax.set_title(f"{m['label']}\n(cols {start}–{start + W - 1})", fontsize=9)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9)
            if r == len(panels) - 1:
                ax.set_xlabel("X index (windowed)")
    fig.colorbar(axes[-1, -1].images[0], ax=list(axes.ravel()), shrink=0.5, pad=0.02,
                 label="velocity / max")
    fig.suptitle("Both velocity components near the seam: in the halo the vector fold is the "
                 "sign-flipped\nscalar fold — the colours invert — for u and v alike. The 180° "
                 "pivot flips velocities; a scalar stays.", fontsize=12, y=1.0)
    plt.show()


def seam_transect(models, K=6, ncols=4):
    '''Continue surface speed (a scalar) across the seam into the halo. The fold
    fills the halo with the true seam-partner row, continuing the field; the
    naive boundary just repeats the edge value (a flat line). A few ocean
    columns per model.'''
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))
    axes = np.atleast_1d(axes)
    for k, (ax, m) in enumerate(zip(axes, models)):
        S = speed_centre(m, True)
        ny = S.sizes["y_c"]
        Sf = _pad_scalar(S, m, K, "fold")
        Se = _pad_scalar(S, m, K, "extend")
        x = np.arange(ny - K, ny + K)
        approach = np.isfinite(Sf[ny - K:ny]).all(axis=0)
        nfin = np.isfinite(Sf[ny - K:ny + K]).sum(axis=0)
        good = np.where(approach & (nfin >= K + 2))[0]
        if good.size == 0:                       # coarse, land-locked cap: most-finite cols
            good = np.argsort(nfin)[::-1][:ncols]
        sel = good[np.linspace(0, len(good) - 1, min(ncols, len(good))).astype(int)]
        for j, i in enumerate(sel):
            lab = (j == 0 and k == 0)
            ax.plot(x, Se[ny - K:ny + K, i], "o--", color="C1", ms=3, alpha=.8,
                    label="naive (extend)" if lab else None)
            ax.plot(x, Sf[ny - K:ny + K, i], "o-", color="C0", ms=3, alpha=.9,
                    label="fold" if lab else None)
        ax.axvline(ny - 0.5, color="k", ls=":", alpha=.6, label="seam" if k == 0 else None)
        ax.set_title(m["label"], fontsize=10)
        ax.set_xlabel("Y index  (interior → halo)")
        if k == 0:
            ax.set_ylabel("surface speed [m s$^{-1}$]")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Across the seam the fold continues the real field; the naive boundary flatlines",
                 fontsize=12)
    plt.tight_layout()
    plt.show()


def div_strip(models, K=6, W=28):
    '''Horizontal divergence ∇·u from `diff` near the seam, formatted exactly like
    the speed strip: naive halo / fold halo / difference. Its ∂v/∂y term crosses
    the fold (using the vector fold of v), so this tests `diff` across the seam.
    Divergence is a TRUE SCALAR, so — like speed, and unlike vorticity — its fold
    halo simply mirrors the interior and continues the field smoothly across the
    seam, while the naive `extend` halo smears the edge. The difference is zero in
    the interior and nonzero only in the halo: the two boundaries differ only
    above the seam, leaving the real field below the line untouched.'''
    rlab = ["naive halo\n(extend)", "fold halo\n(mirror)", "naive − fold"]
    fig, axes = plt.subplots(3, len(models), figsize=(4.6 * len(models), 9.4))

    def lim(a):
        v = np.abs(a[np.isfinite(a)])
        return (float(np.nanpercentile(v, 98)) if v.size else 1.0) or 1.0

    for c, m in enumerate(models):
        D = divergence(m, True)
        ny, nx = D.sizes["y_c"], D.sizes["x_c"]
        start, cols, W = m["win"]
        Df = _pad_scalar(D, m, K, "fold")[ny - K:ny + K]         # fold halo = mirrored interior
        De = _pad_scalar(D, m, K, "extend")[ny - K:ny + K]       # smeared halo
        vmax = lim(np.asarray(D.values)[ny - K:ny])
        dd = (De - Df) / vmax
        div = plt.get_cmap("RdBu_r").copy(); div.set_bad(LAND)
        for r, arr in [(0, De / vmax), (1, Df / vmax), (2, dd)]:
            ax = axes[r, c]
            _imshow(ax, arr[:, cols], ny - K, cmap=div, vmin=-1, vmax=1)
            ax.axhline(ny - 0.5, color="k", lw=1.6)              # the fold seam
            if r == 0:
                ax.set_title(f"{m['label']}\n(cols {start}–{start + W - 1})", fontsize=9)
            if c == 0:
                ax.set_ylabel(rlab[r], fontsize=9)
            if r == 2:
                ax.set_xlabel("X index (windowed)")
    for r, lab in [(0, "∇·u / max"), (1, "∇·u / max"), (2, "(naive−fold) / max")]:
        fig.colorbar(axes[r, -1].images[0], ax=list(axes[r, :]), shrink=0.7, pad=0.02, label=lab)
    fig.suptitle("Horizontal divergence ∇·u from `diff` near the seam (per-model scale): its ∂v/∂y "
                 "term\ncrosses the fold, so this tests `diff` across the seam. Divergence is a true "
                 "scalar — the\nfold halo mirrors the interior and continues the field; naive 'extend' "
                 "smears the edge.", fontsize=11, y=0.99)
    plt.show()
"""),
    md(r"""
## Load the three models

CMIP6 surface velocities for MOM6 (GFDL-CM4) and NEMO (IPSL-CM6A-LR) from the
Pangeo cloud, plus a realistic 1° ClimaOcean/Oceananigans surface snapshot.
CMIP6 masks its redundant northern row, so we drop it before folding.

One subtlety of the CMIP6 tripolar output: the **top row of the meridional
velocity `v`** sits on the fold seam, where its faces are fold duplicates
(`v(i) = -v(mirror(i))`), and the product masks one half of that row. xgcm's fold
fills *halos*, not interior gaps, so `fill_seam_vrow` rebuilds that seam row from
its fold partner at load time — otherwise interpolating `v` to cell centres
(for surface speed) would leave a blank band right at the seam. The fold method
itself doesn't depend on this; it only ever reflects fully-defined interior
rows. (It's a no-op for the Oceananigans field, which is complete.)
"""),
    code(r"""
def _cmip6_surface(source_id, version, fold, label):
    inst = {"GFDL-CM4": "NOAA-GFDL", "IPSL-CM6A-LR": "IPSL"}[source_id]
    base = (f"gs://cmip6/CMIP6/CMIP/{inst}/{source_id}/historical/"
            f"r1i1p1f1/Omon/{{var}}/gn/{version}/")

    def s(var):
        d = xr.open_dataset(base.format(var=var), engine="zarr", backend_kwargs=so)
        return d[var].isel(time=0).isel({d[var].dims[1]: 0})

    uo = s("uo").isel(y=slice(0, -1))
    vo = s("vo").isel(y=slice(0, -1))
    g = xr.open_dataset(base.format(var="uo"), engine="zarr", backend_kwargs=so)
    lonn = "lon" if "lon" in g.variables else "nav_lon"
    latn = "lat" if "lat" in g.variables else "nav_lat"
    return package(uo, vo, g[lonn].isel(y=slice(0, -1)).values,
                   g[latn].isel(y=slice(0, -1)).values, fold, label)


def _oceananigans():
    o = xr.open_dataset("oceananigans_tripolar.nc")
    return package(o["u"].transpose("y_c", "x_f"), o["v"].transpose("y_f", "x_c"),
                   o["lon_cc"].transpose("y_c", "x_c"), o["lat_cc"].transpose("y_c", "x_c"),
                   "u", "Oceananigans (ClimaOcean 1°)")


models = [
    _cmip6_surface("GFDL-CM4", "v20180701", "corner", "MOM6 (GFDL-CM4)"),
    _cmip6_surface("IPSL-CM6A-LR", "v20180803", "corner", "NEMO (IPSL-CM6A-LR)"),
    _oceananigans(),
]
for m in models:          # F-pivot models ship the redundant top v-row masked; rebuild it
    if m["fold"] in ("corner", "f"):
        fill_seam_vrow(m)
attach_windows(models)    # one shared open-water window per model, used by every figure below
"""),
    md(r"""
## `interp` across the fold — what fills the halo

Surface speed at tracer (centre) points near the seam, zoomed to a patch of open
water so each grid cell is visible. The **interior** (below the black line) is
the real field. The **fold halo** above the line is the genuine cross-seam
neighbourhood (the interior reflected about the pole) — real, structured data
that continues the field. The **naive** `extend` halo instead copies the edge
value straight up, so each column is a constant **vertical streak**. The
**difference** is zero in the interior and nonzero only in the halo: the fold
changes nothing inside, it only supplies a physically correct neighbourhood
beyond the edge. (That the continuation is *smooth* is shown by the transect
below. The NEMO panel is gap-free here because we rebuilt its masked seam `v`
row at load time — see above.)
"""),
    code(r"""
seam_strip(models)
"""),
    md(r"""
## Vector components flip sign across the seam

Folding the grid rotates the local axes by 180°, so **both** velocity components
reverse sign across the fold while a scalar does not. To see it cell-by-cell we
fold each component two ways in the same window: as a plain **scalar** and **as
a vector** (passing the other component via `other_component`). For `v`
(meridional) and `u` (zonal) alike, below the seam line the two folds are
identical; in the **halo** above the line they are the same magnitude but
**opposite sign** — the colours invert. That sign flip, on both components, is
the signature of correct vector folding.
"""),
    code(r"""
component_strip(models)
"""),
    md(r"""
## Continuity across the seam — a transect

The cleanest correctness check: follow a few ocean columns from the interior,
across the seam (dotted), into the halo. The **fold** continues the real,
varying field (the seam partner is a genuine physical neighbour); the **naive**
boundary flatlines at the edge value. Where a column's halo runs into Arctic
land the line simply stops — there is no data there, but that is land, not a
fold error.
"""),
    code(r"""
seam_transect(models)
"""),
    md(r"""
## `diff` across the fold — horizontal divergence $\nabla\!\cdot\mathbf{u}$

The same holds for differencing. We use the horizontal divergence
$\nabla\!\cdot\mathbf{u}=\partial u/\partial x+\partial v/\partial y$ at the cell
**centre**, whose $\partial v/\partial y$ term crosses the seam (and needs the
vector fold of `v`), so computing it exercises `diff` across the fold. We choose
divergence deliberately: it is a **true scalar** (invariant under the fold's
180° rotation), so — exactly like surface speed, and *unlike* relative vorticity
$\zeta=\partial v/\partial x-\partial u/\partial y$, which is a sign-flipping
pseudoscalar evaluated at the awkward cell corner — its halo is simply the
mirrored interior, with no sign subtlety and no registration ambiguity. Shown the
same way as the speed strip: the **fold halo** continues the field smoothly
across the seam, the **naive** `extend` halo smears the edge, and their
**difference** is zero in the interior and nonzero only in the halo — the two
boundaries differ only above the seam, leaving the real field below untouched.
"""),
    code(r"""
div_strip(models)
"""),
    md(r"""
## Takeaway

Seen directly in grid-index space, the bipolar north fold is transparent: xgcm
fills the northern halo with the interior **reflected about the poles**,
sign-flipping vector components. So

* a smooth scalar (surface speed) **continues across the seam** — the fold halo
  is the mirrored interior, the transect stays continuous — while the naive
  `extend` boundary smears the edge and flatlines;
* a velocity component additionally **reverses sign** across the seam, the
  signature of the 180° pivot;
* a `diff`-based true scalar (the horizontal divergence) **continues across the
  seam** just like an interpolated scalar — corrected **exactly along the seam
  row** and left untouched elsewhere — confirming `diff` works across the fold.

Because the halo is provably the reflected interior, any wiggle at the seam — for
instance the grid-scale noise in the coarse, short-spin-up Oceananigans field —
is the **model's own velocity field, faithfully mirrored**, not an artifact of
the fold operators. The mirror symmetry is the tell. xgcm's
`boundary={"X": "periodic", "Y": {"fold": ...}}` makes the standard staggered
`interp`/`diff`/`derivative` work across the pole for all three models'
conventions (`"corner"` for MOM6/NEMO, `"u"` for Oceananigans).

See the [grid topology](../grid_topology.md) docs for the four fold pivots and
how the halo is filled, and [`03_MOM6.ipynb`](03_MOM6.ipynb) for more MOM6
recipes.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(EX, "06_tripolar_fold.ipynb"), "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote 06_tripolar_fold.ipynb")
