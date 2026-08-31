"""
model_currents.py
=================
Read tidal-current harmonic constituents from the user's FES2022 and TPXO10
netCDF files, interpolate them to a point, and reconstruct the depth-averaged
current-speed time series. The harmonic synthesis (frequencies, astronomical
arguments and nodal corrections) is done with utide's validated engine, so no
pyTMD/pyfes installation is needed -- only utide, xarray and netCDF4, which the
main pipeline already uses.

File layouts expected (exactly as downloaded):

  TPXO10  (one folder, transports + grid):
      <TPXO_DIR>/u_<con>_tpxo10_atlas_30_v2.nc   (uRe,uIm,vRe,vIm transports)
      <TPXO_DIR>/grid_tpxo10atlas_v2.nc          (depth, for transport->velocity)

  FES2022 (two folders, velocity amplitude+phase):
      <FES_DIR>/eastward_velocity/<con>_fes2022.nc
      <FES_DIR>/northward_velocity/<con>_fes2022.nc

The readers auto-detect variable and coordinate names and print what they find,
so if a file uses different names the diagnostic makes the fix obvious. Because
the downstream method calibrates amplitude to local data (ABPmer at MeyGen, MD6.1
at EMEC), any residual unit-scale ambiguity in the absolute currents cancels for
the calibrated sites; the temporal STRUCTURE (relative amplitudes, phases, the
spring-neap and monthly pattern) is what the models supply and it is unit-free.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from utide.harmonics import ut_E, ut_constants

_NAMES = [n.strip() for n in ut_constants["const"]["name"]]
_FREQ = ut_constants["const"]["freq"]

# file token (lower-case, before _fes2022 / between u_ and _tpxo10) -> utide name
_CON_MAP = {
    "2n2": "2N2", "eps2": "EPS2", "j1": "J1", "k1": "K1", "k2": "K2", "l2": "L2",
    "lambda2": "LDA2", "la2": "LDA2", "m2": "M2", "m3": "M3", "m4": "M4", "m6": "M6",
    "m8": "M8", "mf": "MF", "mks2": "MKS2", "mm": "MM", "mn4": "MN4", "ms4": "MS4",
    "msf": "MSF", "mu2": "MU2", "n2": "N2", "n4": "N4", "nu2": "NU2", "o1": "O1",
    "p1": "P1", "q1": "Q1", "r2": "R2", "s1": "S1", "s2": "S2", "s4": "S4",
    "sa": "SA", "ssa": "SSA", "t2": "T2",
    # long-period constituents absent from utide's table are skipped:
    "msqm": None, "mtm": None,
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _datenum(times):
    idx = pd.to_datetime(times)
    return np.array([d.toordinal() + d.hour / 24 + d.minute / 1440 + d.second / 86400
                     for d in idx])


def _pick_coord(ds, axis):
    """Pick the lon/lat coordinate deterministically: skip bounds/vertices,
    prefer a 1-D variable and the shortest name (so 'lon' beats 'lon_bnds')."""
    cands = [c for c in list(ds.coords) + list(ds.variables)
             if axis in c.lower()
             and not any(b in c.lower() for b in ("bnd", "bound", "vert", "_b"))]
    cands = sorted(set(cands), key=lambda c: (np.asarray(ds[c].values).ndim, len(c)))
    return cands[0]


def _as_latlon(arr, LON, LAT):
    """Orient a 2-D field as (lat, lon)."""
    if arr.shape == (len(LAT), len(LON)):
        return arr
    if arr.shape == (len(LON), len(LAT)):
        return arr.T
    return arr


def _amp_pha_vars(ds):
    """Identify the amplitude and phase variables robustly: amplitude has velocity
    units (m/s, cm/s) or an 'a'/'amp' name (Ua, Va, amplitude); phase has angle
    units (degrees) or a 'g'/'pha' name (Ug, Vg, phase). Works for both the
    eastward (Ua/Ug) and northward (Va/Vg) FES files."""
    amp = pha = None
    for v in ds.data_vars:
        lv = v.lower()
        if any(b in lv for b in ("bnd", "bound", "crs", "lon", "lat", "vert")):
            continue
        u = str(ds[v].attrs.get("units", "")).lower()
        if amp is None and ("m/s" in u or "m s" in u or "cm/s" in u or "cm s" in u
                            or "amp" in lv or lv in ("ua", "va")):
            amp = v
        elif pha is None and ("deg" in u or "phase" in lv or "pha" in lv or lv in ("ug", "vg")):
            pha = v
    cand = [v for v in ds.data_vars
            if not any(b in v.lower() for b in ("bnd", "bound", "crs", "lon", "lat", "vert"))]
    if amp is None:
        amp = cand[0]
    if pha is None:
        pha = next((v for v in cand if v != amp), amp)
    return amp, pha


def _grid(ds):
    return (np.asarray(ds[_pick_coord(ds, "lon")].values),
            np.asarray(ds[_pick_coord(ds, "lat")].values))


def _nearest_ij(LON, LAT, finite, lon, lat):
    """(lat, lon) index of the nearest cell where `finite` is True (skips land)."""
    qlon = lon % 360 if np.nanmax(LON) > 180 and lon < 0 else lon
    if LON.ndim == 1 and LAT.ndim == 1:
        dlon = (np.abs(((LON - qlon + 180) % 360) - 180)) ** 2
        D = (LAT - lat)[:, None] ** 2 + dlon[None, :]
    else:
        D = (LON - qlon) ** 2 + (LAT - lat) ** 2
    D = np.where(finite, D, np.inf)
    if not np.isfinite(D).any():
        return None
    return np.unravel_index(int(np.argmin(D)), D.shape)


def _interp(ds, varname, lon, lat):
    """Nearest-VALID-grid-point value of `varname` at (lon, lat)."""
    LON, LAT = _grid(ds)
    arr = _as_latlon(np.asarray(ds[varname].values).squeeze(), LON, LAT)
    ij = _nearest_ij(LON, LAT, np.isfinite(arr), lon, lat)
    return np.nan if ij is None else float(arr[ij])


def _find(ds, *keys):
    """Return the first data-variable name whose lowercase contains all keys."""
    for v in ds.data_vars:
        lv = v.lower()
        if all(k in lv for k in keys):
            return v
    return None


# ----------------------------------------------------------------------------
# TPXO10: transports (cm^2/s) -> velocity (m/s) via grid depth
# ----------------------------------------------------------------------------
def _tpxo_depth(tpxo_dir, lon, lat):
    grid = list(Path(tpxo_dir).glob("grid_*tpxo10*"))[0]
    with xr.open_dataset(grid) as ds:
        dvar = _find(ds, "hz") or _find(ds, "depth") or _find(ds, "h")
        LON, LAT = _grid(ds)
        arr = _as_latlon(np.abs(np.asarray(ds[dvar].values).squeeze()), LON, LAT)
        ij = _nearest_ij(LON, LAT, np.isfinite(arr) & (arr > 1.0), lon, lat)   # nearest WET cell
        d = float(arr[ij]) if ij is not None else np.nan
    return d if (np.isfinite(d) and d > 1) else np.nan


def _tpxo_constituent(tpxo_dir, token, lon, lat, depth, verbose=False):
    f = list(Path(tpxo_dir).glob(f"u_{token}_tpxo10*"))
    if not f or not np.isfinite(depth):
        return None
    with xr.open_dataset(f[0]) as ds:
        uRe, uIm = _find(ds, "ure") or _find(ds, "u", "re"), _find(ds, "uim") or _find(ds, "u", "im")
        vRe, vIm = _find(ds, "vre") or _find(ds, "v", "re"), _find(ds, "vim") or _find(ds, "v", "im")
        if verbose:
            print(f"    TPXO {token}: vars {uRe},{uIm},{vRe},{vIm}")
        LON, LAT = _grid(ds)
        URe = _as_latlon(np.asarray(ds[uRe].values).squeeze(), LON, LAT)
        UIm = _as_latlon(np.asarray(ds[uIm].values).squeeze(), LON, LAT)
        VRe = _as_latlon(np.asarray(ds[vRe].values).squeeze(), LON, LAT)
        VIm = _as_latlon(np.asarray(ds[vIm].values).squeeze(), LON, LAT)
        magu = np.hypot(URe, UIm)                            # u wet where != 0
        magv = np.hypot(VRe, VIm)                            # v wet where != 0 (staggered grid)
        iju = _nearest_ij(LON, LAT, np.isfinite(magu) & (magu > 0), lon, lat)
        ijv = _nearest_ij(LON, LAT, np.isfinite(magv) & (magv > 0), lon, lat)
        if iju is None or ijv is None:
            return None
        uc = complex(URe[iju], UIm[iju])
        vc = complex(VRe[ijv], VIm[ijv])
    # transports cm^2/s -> velocity m/s: divide by depth and convert (cm^2/s)/(m) -> m/s
    scale = 1.0 / (depth * 1.0e4)          # cm^2/s / (m) * 1e-4 -> m/s
    uc, vc = uc * scale, vc * scale
    Au, gu = abs(uc), np.degrees(np.arctan2(-uc.imag, uc.real)) % 360
    Av, gv = abs(vc), np.degrees(np.arctan2(-vc.imag, vc.real)) % 360
    return Au, gu, Av, gv


# ----------------------------------------------------------------------------
# FES2022: eastward/northward velocity amplitude (cm/s) + phase (deg)
# ----------------------------------------------------------------------------
def _fes_dir_constituent(folder, token, lon, lat):
    f = list(Path(folder).glob(f"{token}_fes2022.nc"))
    if not f:
        return None
    with xr.open_dataset(f[0]) as ds:
        amp_v, pha_v = _amp_pha_vars(ds)
        LON, LAT = _grid(ds)
        A = _as_latlon(np.asarray(ds[amp_v].values).squeeze(), LON, LAT)
        P = _as_latlon(np.asarray(ds[pha_v].values).squeeze(), LON, LAT)
        ij = _nearest_ij(LON, LAT, np.isfinite(A), lon, lat)
        if ij is None:
            return None
        A_val, g_val = float(A[ij]), float(P[ij])
        units = str(ds[amp_v].attrs.get("units", "")).lower()
    if not np.isfinite(A_val):
        return None
    return A_val, g_val % 360, units


def _fes_constituent(fes_dir, token, lon, lat):
    e = _fes_dir_constituent(Path(fes_dir) / "eastward_velocity", token, lon, lat)
    n = _fes_dir_constituent(Path(fes_dir) / "northward_velocity", token, lon, lat)
    if e is None or n is None:
        return None
    return e[0], e[1], n[0], n[1], e[2]    # Au, gu, Av, gv, units (raw, unscaled)


# ----------------------------------------------------------------------------
# reconstruction (validated against utide)
# ----------------------------------------------------------------------------
def _reconstruct(con, times, lat):
    """con: {utide_name: (Au, gu, Av, gv)} -> depth-averaged speed series [m/s]."""
    names = [n for n in con if n in _NAMES]
    lind = np.array([_NAMES.index(n) for n in names])
    frq = np.array([_FREQ[i] for i in lind])
    t = _datenum(times)
    tref = t.mean()
    E = ut_E(t, tref, frq, lind, lat, [0, 0, 0, 0], None)
    rpd = np.pi / 180
    Xu = np.array([con[n][0] * np.exp(-1j * con[n][1] * rpd) for n in names])
    Xv = np.array([con[n][2] * np.exp(-1j * con[n][3] * rpd) for n in names])
    ap = 0.5 * (Xu + 1j * Xv)
    am = 0.5 * (np.conj(Xu) + 1j * np.conj(Xv))
    w = E @ ap + np.conj(E) @ am
    return pd.Series(np.hypot(w.real, w.imag), index=pd.to_datetime(times))


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def _collect_constituents(lon, lat, source, tpxo_dir=None, fes_dir=None, verbose=False):
    """Return {utide_name: (Au, gu, Av, gv)} in m/s at (lon, lat) for one model."""
    con = {}
    if source.upper().startswith("TPXO"):
        depth = _tpxo_depth(tpxo_dir, lon, lat)
        for tok, nm in _CON_MAP.items():
            if nm is None:
                continue
            r = _tpxo_constituent(tpxo_dir, tok, lon, lat, depth, verbose)
            if r is not None:
                con[nm] = r
    else:
        if verbose:
            for sub in ("eastward_velocity", "northward_velocity"):
                f0 = list((Path(fes_dir) / sub).glob("m2_fes2022.nc"))
                if f0:
                    with xr.open_dataset(f0[0]) as d0:
                        a_v, p_v = _amp_pha_vars(d0)
                        ex = _fes_dir_constituent(Path(fes_dir) / sub, "m2", lon, lat)
                        print(f"    FES {sub}: amp='{a_v}' phase='{p_v}' "
                              f"-> M2 amp {ex[0] if ex else 'NaN'}")
        raw = {}
        for tok, nm in _CON_MAP.items():
            if nm is None:
                continue
            r = _fes_constituent(fes_dir, tok, lon, lat)
            if r is not None:
                raw[nm] = r
        if not raw:
            raise RuntimeError("No FES constituents read; check paths/variable names.")
        units = raw.get("M2", next(iter(raw.values())))[4]
        if "cm" in units:
            scale = 0.01
        elif "m/s" in units or "m s" in units:
            scale = 1.0
        else:                               # unknown units: infer from M2 magnitude
            m2amp = max(raw["M2"][0], raw["M2"][2]) if "M2" in raw else 1.0
            scale = 0.01 if m2amp > 10 else 1.0
        if verbose:
            print(f"    FES2022 amplitude units '{units or 'unknown'}' -> scale x{scale}")
        for nm, (Au, gu, Av, gv, _u) in raw.items():
            con[nm] = (Au * scale, gu, Av * scale, gv)
    if not con:
        raise RuntimeError(f"No constituents read for {source}; check paths/variable names.")
    if verbose:
        print(f"    {source}: {len(con)} constituents -> {', '.join(con)}")
    return con


def _reconstruct_uv(con, times, lat):
    """con -> (u, v) depth-averaged component series [m/s] (for EDA / ellipses)."""
    names = [n for n in con if n in _NAMES]
    lind = np.array([_NAMES.index(n) for n in names])
    frq = np.array([_FREQ[i] for i in lind])
    t = _datenum(times)
    E = ut_E(t, t.mean(), frq, lind, lat, [0, 0, 0, 0], None)
    rpd = np.pi / 180
    Xu = np.array([con[n][0] * np.exp(-1j * con[n][1] * rpd) for n in names])
    Xv = np.array([con[n][2] * np.exp(-1j * con[n][3] * rpd) for n in names])
    ap = 0.5 * (Xu + 1j * Xv)
    am = 0.5 * (np.conj(Xu) + 1j * np.conj(Xv))
    w = E @ ap + np.conj(E) @ am
    return w.real, w.imag


def model_speed(lon, lat, times, source, tpxo_dir=None, fes_dir=None, verbose=False):
    """Depth-averaged tidal-current speed [m/s] at (lon, lat) for `times`,
    from 'FES2022' (fes_dir) or 'TPXO10' (tpxo_dir)."""
    con = _collect_constituents(lon, lat, source, tpxo_dir, fes_dir, verbose)
    return _reconstruct(con, times, lat)


def eda_plots(lon, lat, source, tpxo_dir=None, fes_dir=None, label=None, outdir="figs/EDA"):
    """Exploratory data analysis of the model currents at one point: constituent
    amplitude spectrum, tidal-current ellipse, reconstructed spring-neap envelope
    and speed-exceedance curve. Makes the harmonic reconstruction transparent."""
    import matplotlib.pyplot as plt
    from pathlib import Path as _P
    label = label or f"{source}"
    con = _collect_constituents(lon, lat, source, tpxo_dir, fes_dir)
    times = pd.date_range("2024-01-01", "2024-02-19", freq="20min")   # ~50 days
    u, v = _reconstruct_uv(con, times, lat)
    spd = np.hypot(u, v)
    # constituent amplitudes |A| = hypot(Au, Av), sorted
    amp = {k: float(np.hypot(c[0], c[2])) for k, c in con.items()}
    top = sorted(amp.items(), key=lambda kv: -kv[1])[:12]
    names, vals = [t[0] for t in top], [t[1] for t in top]

    _P(outdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"EDA — {source} depth-averaged currents at {label} "
                 f"(lat {lat:.3f}, lon {lon:.3f})", fontsize=12)

    # (a) constituent amplitude spectrum
    ax[0, 0].bar(range(len(names)), vals, color="#356", alpha=0.85)
    ax[0, 0].set_xticks(range(len(names))); ax[0, 0].set_xticklabels(names, rotation=45, fontsize=8)
    ax[0, 0].set_ylabel("Current amplitude |A| [m/s]")
    ax[0, 0].set_title("(a) Constituent spectrum (M2 dominant; S2/N2 set spring-neap)")

    # (b) tidal-current ellipse / hodograph (first ~15 days)
    m = times < times[0] + pd.Timedelta(days=15)
    ax[0, 1].plot(u[m], v[m], lw=0.5, color="#a33", alpha=0.7)
    lim = 1.05 * np.nanmax(np.hypot(u, v))
    ax[0, 1].set_xlim(-lim, lim); ax[0, 1].set_ylim(-lim, lim); ax[0, 1].set_aspect("equal")
    ax[0, 1].axhline(0, lw=0.5, c="k"); ax[0, 1].axvline(0, lw=0.5, c="k")
    ax[0, 1].set_xlabel("Eastward u [m/s]"); ax[0, 1].set_ylabel("Northward v [m/s]")
    ax[0, 1].set_title("(b) Current hodograph (thin ellipse = rectilinear flow)")

    # (c) reconstructed speed, spring-neap envelope
    s = pd.Series(spd, index=times)
    dm = s.resample("D").max()
    ax[1, 0].plot(s.index, s.values, lw=0.5, color="#333")
    ax[1, 0].axhline(dm.quantile(0.95), ls="--", c="#c00", lw=1,
                     label=f"spring ~{dm.quantile(0.95):.2f}")
    ax[1, 0].axhline(dm.quantile(0.25), ls="--", c="#06c", lw=1,
                     label=f"neap ~{dm.quantile(0.25):.2f}")
    ax[1, 0].set_ylabel("Speed [m/s]"); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("(c) Reconstructed speed: spring-neap envelope")
    for lab in ax[1, 0].get_xticklabels():
        lab.set_rotation(20); lab.set_fontsize(7)

    # (d) exceedance curve
    ss = np.sort(spd)[::-1]
    exc = np.arange(1, len(ss) + 1) / len(ss) * 100
    ax[1, 1].plot(exc, ss, color="#356")
    ax[1, 1].set_xlabel("Exceedance [%]"); ax[1, 1].set_ylabel("Speed [m/s]")
    ax[1, 1].set_title(f"(d) Speed-exceedance — mean {spd.mean():.2f} m/s, "
                       f"{(spd > 1).mean()*100:.0f}% of time above 1 m/s")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fn = f"{outdir}/eda_{source}_{(label or '').replace(' ', '_')}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    print(f"  EDA figure -> {fn}")
    return fn


# ----------------------------------------------------------------------------
# smoke test: `python model_currents.py` reads your real files at two sites and
# prints diagnostics (constituent count, FES unit scale, spring/neap peaks).
# Adjust the two paths below if your folders differ.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    TPXO_DIR = Path(r"C:\Users\Steven Tarigan\Tidal dissertation\Download\TPXO10_atlas_v2_nc")
    FES_DIR = Path(r"C:\Users\Steven Tarigan\Tidal dissertation\Download\FES2022B_currents")
    SITES = {"EMEC Fall of Warness": (59.136, -2.801),
             "MeyGen Inner Sound":   (58.658, -3.143),
             "Morlais Anglesey":     (53.310, -4.690)}   # atlas is the ONLY resource here
                                                         # (calib_sp/ne are None in
                                                         # tidal_model_resource.SITES), so the
                                                         # EDA matters most at this site
    times = pd.date_range("2024-01-01", "2024-02-01", freq="30min")
    print("=== model_currents smoke test (depth-averaged speed, m/s) ===")
    for site, (lat, lon) in SITES.items():
        print(f"\n{site}  (lat {lat}, lon {lon})")
        for source, kw in [("FES2022", dict(fes_dir=FES_DIR)),
                           ("TPXO10", dict(tpxo_dir=TPXO_DIR))]:
            try:
                s = model_speed(lon, lat, times, source, verbose=True, **kw)
                dm = s.resample("D").max()
                print(f"  {source:8s} mean {s.mean():.2f}  "
                      f"spring(p95) {dm.quantile(0.95):.2f}  neap(p25) {dm.quantile(0.25):.2f}")
                try:
                    eda_plots(lon, lat, source, label=site, **kw)     # EDA figure
                except Exception as e:                                # noqa: BLE001
                    print(f"    (EDA plot skipped: {type(e).__name__}: {e})")
            except Exception as exc:                                  # noqa: BLE001
                print(f"  {source:8s} ERROR: {type(exc).__name__}: {exc}")
    print("\n(depth-averaged; multiply by HUB_RATIO ~1.32 for hub height. "
          "EDA figures written to ./figs/EDA/. If a model errors, the printed "
          "variable names show what to adjust.)")
