#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API:
- POST /api/compute -> Berechnet Niederschlagsdaten für einen bestimmten Monat
- POST /api/range -> Gibt Niederschlagsdaten für einen bestimmten Zeitraum zurück
- GET /r/<job_id>/calendar.png -> Ruft das Kalender-PNG für einen bestimmten Job ab
- GET /r/<job_id>/bars.png -> Ruft das Balken-PNG für einen bestimmten Job ab
- GET /r/<job_id>/cumulative.png -> Ruft das kumulative PNG für einen bestimmten Job ab
- GET /r/<job_id>/daily.csv -> Ruft die tägliche CSV-Datei für einen bestimmten Job ab
- GET /r/<job_id>/result.json -> Ruft die Ergebnis-JSON-Datei für einen bestimmten Job ab
- GET /r/<job_id>/range.csv -> Ruft die Bereichs-CSV-Datei für einen bestimmten Job ab
- GET /r/<job_id>/range.json -> Ruft die Bereichs-JSON-Datei für einen bestimmten Job ab
- GET /healthz -> Überprüft den Zustand des Dienstes
"""
"""
fl-dwd-precip-calendar — DWD Niederschlag als Tages-Kalenderplot (AOI-Mittel)

Datenquelle (empfohlen):
- DWD CDC HYRAS-DE-PR v6-1 (daily NetCDF): 1 km Raster, Tagessummen in mm, EPSG:3035,
  aktuelles Jahr wird täglich um einen Tag erweitert. (Monat kann unvollständig sein.)

Warum nicht WFS?
- WFS = Vektorfeatures (z.B. Stationen/Polygone). Für flächiges Mittel brauchst du Rasterwerte
  (WCS oder Datei: NetCDF/GeoTIFF). WMS liefert nur Bilder.

Cloud Run:
- Single-file Flask app, nutzt /tmp Cache.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import platform
import tempfile

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

import matplotlib
matplotlib.use("Agg")  # server backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from affine import Affine
from rasterio.features import rasterize

try:
    from netCDF4 import Dataset, num2date  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency netCDF4. Install: pip install netCDF4") from e


# -------------------------------
# Config (Cloud Run friendly)
# -------------------------------

APP_TITLE = os.getenv("APP_TITLE", "fl-dwd-precip-calendar (HYRAS) – Tagesniederschlag als Kalenderplot")
SERVICE_SLUG = os.getenv("SERVICE_SLUG", "fl-dwd-precip-calendar")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))

# HYRAS daily precipitation (directory listing)
HYRAS_DAILY_BASE = os.getenv(
    "HYRAS_DAILY_BASE",
    "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/",
)

# Safety
MAX_AOI_AREA_KM2 = float(os.getenv("MAX_AOI_AREA_KM2", "25.0"))
MAX_SUBSET_PIXELS = int(os.getenv("MAX_SUBSET_PIXELS", "2200000"))  # ~2.2 Mio for safety
MAX_RANGE_DAYS = int(os.getenv("MAX_RANGE_DAYS", "400"))  # safety for range exports (inclusive)
DEFAULT_MONTH = int(os.getenv("DEFAULT_MONTH", "0"))  # 0 => current month

# Cache
def _default_cache_dir() -> Path:
    """
    Windows: %TEMP% (tempfile.gettempdir) statt /tmp.
    Linux/Cloud Run: /tmp ist ok.
    Fallback: ./data/cache falls TEMP nicht beschreibbar ist.
    """
    env = os.getenv("TMP_DIR")
    if env:
        base = Path(env)
    else:
        if platform.system() == "Windows":
            base = Path(tempfile.gettempdir())
        else:
            base = Path("/tmp")

    p = base / "hyras_precip_cache"
    try:
        p.mkdir(parents=True, exist_ok=True)
        # write test
        t = p / ".write_test"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
        return p
    except Exception:
        p2 = Path.cwd() / "data" / "cache" / "hyras_precip_cache"
        p2.mkdir(parents=True, exist_ok=True)
        return p2

TMP_DIR = _default_cache_dir()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "12"))  # e.g. a few yearly nc + products

# HYRAS projection (from dataset description)
HYRAS_EPSG = 3035


# -------------------------------
# Flask
# -------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_AS_ASCII"] = False


@app.after_request
def _add_headers(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# -------------------------------
# Data classes
# -------------------------------

@dataclass
class ComputeResult:
    job_id: str
    year: int
    month: int
    month_name: str
    aoi_area_km2: float
    last_available_date: str
    series: List[Dict[str, Any]]
    stats: Dict[str, Any]
    calendar_png: Path
    bars_png: Path
    cumulative_png: Path
    csv_path: Path
    json_path: Path


# -------------------------------
# Helpers (cache, geojson, http)
# -------------------------------

def _now_ts() -> float:
    return time.time()


def _cleanup_cache() -> None:
    try:
        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)

        now = _now_ts()
        for mtime, p in items:
            if now - mtime > CACHE_TTL_SECONDS:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        items = []
        for p in TMP_DIR.glob("*"):
            if p.is_file():
                items.append((p.stat().st_mtime, p))
        items.sort(reverse=True)

        if len(items) > MAX_CACHE_ITEMS:
            for _, p in items[MAX_CACHE_ITEMS:]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def _json_error(msg: str, status: int = 400, **details: Any):
    payload = {"error": msg}
    if details:
        payload["details"] = details
    return jsonify(payload), status


def _parse_geojson(payload: Any) -> Dict[str, Any]:
    if payload is None:
        raise ValueError("Kein GeoJSON übergeben.")
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError("Leerer GeoJSON-String.")
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("GeoJSON muss JSON-Objekt oder String sein.")


def _extract_single_geometry(gj: Dict[str, Any]):
    t = gj.get("type")
    if t == "Feature":
        geom = gj.get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t == "FeatureCollection":
        feats = gj.get("features") or []
        if len(feats) != 1:
            raise ValueError("FeatureCollection muss genau 1 Feature enthalten.")
        geom = feats[0].get("geometry")
        if not geom:
            raise ValueError("Feature ohne geometry.")
        return shape(geom)
    if t in ("Polygon", "MultiPolygon"):
        return shape(gj)
    raise ValueError(f"Nicht unterstützter GeoJSON-Typ: {t}. Erlaubt: Feature, FeatureCollection(1), Polygon, MultiPolygon.")


def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def _geom_to_epsg(geom, src_epsg: int, dst_epsg: int):
    tr = _transformer(src_epsg, dst_epsg)
    return shp_transform(lambda x, y: tr.transform(x, y), geom)


def _http_get_text(url: str, tries: int = 3) -> str:
    last_err = None
    for _ in range(max(1, tries)):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": f"{SERVICE_SLUG}/1.0 (+https://data-tales.dev/)"} )
            txt = r.text or ""
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {txt[:500]}")
            return txt
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"HTTP GET failed: {last_err}")


def _http_download_file(url: str, out_path: Path, tries: int = 3) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # if cached and recent -> keep
    if out_path.exists():
        age = _now_ts() - out_path.stat().st_mtime
        if age <= CACHE_TTL_SECONDS:
            return out_path

    last_err = None
    for _ in range(max(1, tries)):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT, headers={"User-Agent": f"{SERVICE_SLUG}/1.0 (+https://data-tales.dev/)"} ) as r:
                if not r.ok:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(out_path)
                if not out_path.exists() or out_path.stat().st_size < 1024:
                    raise RuntimeError(f"Download unvollständig: {out_path}")

                return out_path
        except Exception as e:
            last_err = e
            try:
                if out_path.with_suffix(out_path.suffix + ".part").exists():
                    out_path.with_suffix(out_path.suffix + ".part").unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(0.8)
    raise RuntimeError(f"Download failed: {last_err}")


def _find_hyras_daily_file(year: int) -> str:
    """
    Picks the best matching HYRAS daily NetCDF file in the directory.
    Expected patterns (examples):
      pr_hyras_1_2025_v6-1_de.nc
      pr_hyras_1_2025_v6-0_de.nc
    We parse directory listing and prefer the highest version.
    """
    html = _http_get_text(HYRAS_DAILY_BASE, tries=3)

    # collect candidate filenames
    rx = re.compile(r'href="(pr_hyras_1_' + re.escape(str(year)) + r'_v[\d\-]+_de\.nc)"', re.IGNORECASE)
    cands = rx.findall(html) or []
    if not cands:
        # fallback: maybe without quotes or different listing format
        rx2 = re.compile(r"(pr_hyras_1_" + re.escape(str(year)) + r"_v[\d\-]+_de\.nc)", re.IGNORECASE)
        cands = list(set(rx2.findall(html) or []))

    if not cands:
        raise ValueError(f"Kein HYRAS daily NetCDF für Jahr {year} gefunden in: {HYRAS_DAILY_BASE}")

    # sort: prefer v6-1 over v6-0 etc
    def _ver_key(fn: str) -> Tuple[int, int]:
        m = re.search(r"_v(\d+)-(\d+)_", fn)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))

    cands = sorted(set(cands), key=_ver_key, reverse=True)
    return cands[0]


def _month_name_de(m: int) -> str:
    names = ["Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"]
    return names[m-1] if 1 <= m <= 12 else str(m)


# -------------------------------
# NetCDF reading + AOI mean
# -------------------------------

def _pick_precip_variable(nc: Dataset):
    """
    Choose a 3D variable (time,y,x) that looks like precipitation.
    HYRAS typically uses a variable with units 'mm' (or similar).
    """
    vars3d = []
    for name, v in nc.variables.items():
        try:
            dims = getattr(v, "dimensions", ())
            if len(dims) != 3:
                continue
            vars3d.append((name, v))
        except Exception:
            continue

    if not vars3d:
        raise ValueError("Keine 3D-Variable (time,y,x) im NetCDF gefunden.")

    # prefer units containing "mm"
    for name, v in vars3d:
        units = (getattr(v, "units", "") or "").lower()
        stdn = (getattr(v, "standard_name", "") or "").lower()
        longn = (getattr(v, "long_name", "") or "").lower()
        if "mm" in units or "precip" in stdn or "niederschlag" in longn:
            return name, v

    # common name fallback
    for name, v in vars3d:
        if name.lower() in ("pr", "rr", "precip", "precipitation"):
            return name, v

    return vars3d[0]


def _get_xy(nc: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    # common coordinate names
    for xn, yn in (("x", "y"), ("X", "Y"), ("easting", "northing")):
        if xn in nc.variables and yn in nc.variables:
            x = np.array(nc.variables[xn][:], dtype=np.float64)
            y = np.array(nc.variables[yn][:], dtype=np.float64)
            return x, y
    # fallback: scan for 1D numeric variables with long_name hints
    x = y = None
    for name, v in nc.variables.items():
        if getattr(v, "ndim", 0) != 1:
            continue
        ln = (getattr(v, "long_name", "") or "").lower()
        if "x" == name.lower() or "easting" in ln:
            x = np.array(v[:], dtype=np.float64)
        if "y" == name.lower() or "northing" in ln:
            y = np.array(v[:], dtype=np.float64)
    if x is None or y is None:
        raise ValueError("x/y Koordinatenvariablen im NetCDF nicht gefunden.")
    return x, y


def _subset_slices_from_bounds(x: np.ndarray, y: np.ndarray, bounds: Tuple[float, float, float, float]) -> Tuple[slice, slice]:
    minx, miny, maxx, maxy = bounds

    # include slight margin
    dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1000.0
    dy = float(np.median(np.abs(np.diff(y)))) if y.size > 1 else 1000.0

    xmask = (x >= (minx - dx)) & (x <= (maxx + dx))
    ymask = (y >= (miny - dy)) & (y <= (maxy + dy))

    xi = np.where(xmask)[0]
    yi = np.where(ymask)[0]
    if xi.size == 0 or yi.size == 0:
        raise ValueError("AOI liegt außerhalb des HYRAS Rasters (keine x/y Indizes).")

    xs = slice(int(xi.min()), int(xi.max()) + 1)
    ys = slice(int(yi.min()), int(yi.max()) + 1)
    return ys, xs


def _affine_for_subset(x: np.ndarray, y: np.ndarray, ys: slice, xs: slice) -> Affine:
    x_sub = x[xs]
    y_sub = y[ys]

    if x_sub.size < 2 or y_sub.size < 2:
        raise ValueError("Subset zu klein für Transform-Berechnung.")

    dx = float(np.median(np.diff(x_sub)))
    dy_raw = float(np.median(np.diff(y_sub)))
    dy_abs = float(np.median(np.abs(np.diff(y_sub))))

    left = float(x_sub[0] - dx / 2.0)

    if dy_raw < 0:
        # y descending (north->south), usual north-up raster
        top = float(y_sub[0] + dy_abs / 2.0)
        return Affine(dx, 0.0, left, 0.0, -dy_abs, top)
    else:
        # y ascending (south->north)
        top = float(y_sub[0] - dy_abs / 2.0)
        return Affine(dx, 0.0, left, 0.0, +dy_abs, top)


def _aoi_mask_for_subset(geom_3035, out_shape: Tuple[int, int], transform: Affine) -> np.ndarray:
    h, w = out_shape
    m = rasterize(
        [mapping(geom_3035)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype=np.uint8,
    )
    return (m.astype(np.uint8) == 1)


def _compute_month_series_hyras(gj: Dict[str, Any], year: int, month: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    _cleanup_cache()

    geom_wgs84 = _extract_single_geometry(gj)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    geom_3035 = _geom_to_epsg(geom_wgs84, 4326, HYRAS_EPSG)
    aoi_area_km2 = float(geom_3035.area) / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    # resolve file
    fn = _find_hyras_daily_file(year)
    url = HYRAS_DAILY_BASE.rstrip("/") + "/" + fn
    local_nc = TMP_DIR / fn
    _http_download_file(url, local_nc, tries=3)

    with Dataset(str(local_nc), mode="r") as nc:
        _, vpr = _pick_precip_variable(nc)

        # time
        tvar = None
        for cand in ("time", "TIME", "t"):
            if cand in nc.variables:
                tvar = nc.variables[cand]
                break
        if tvar is None:
            for name, vv in nc.variables.items():
                if getattr(vv, "ndim", 0) == 1:
                    u = (getattr(vv, "units", "") or "").lower()
                    if "since" in u and "day" in u:
                        tvar = vv
                        break
        if tvar is None:
            raise ValueError("Keine Zeitvariable im NetCDF gefunden.")

        t_units = getattr(tvar, "units", None)
        t_cal = getattr(tvar, "calendar", "standard")
        tvals = np.array(tvar[:])

        dts = num2date(tvals, units=t_units, calendar=t_cal)
        dates = [date(dt.year, dt.month, dt.day) for dt in dts]
        date_to_idx = {d: i for i, d in enumerate(dates)}

        last_avail = max(dates) if dates else date(year, month, 1)

        # month day list
        last_day = calendar.monthrange(year, month)[1]
        month_days = [date(year, month, d) for d in range(1, last_day + 1)]

        # subset bounds
        minx, miny, maxx, maxy = geom_3035.bounds

        x, y = _get_xy(nc)
        ys, xs = _subset_slices_from_bounds(x, y, (minx, miny, maxx, maxy))

        y_sub = y[ys]
        x_sub = x[xs]
        h = int(y_sub.size)
        w = int(x_sub.size)

        if h <= 0 or w <= 0:
            raise ValueError("Subset leer (h/w <= 0).")

        if (h * w) > MAX_SUBSET_PIXELS:
            raise ValueError(f"AOI-Subset zu groß: {h*w} Pixel (Limit: {MAX_SUBSET_PIXELS}).")

        transform = _affine_for_subset(x, y, ys, xs)
        mask = _aoi_mask_for_subset(geom_3035, (h, w), transform)

        # compute daily mean over AOI
        out_series: List[Dict[str, Any]] = []
        for d in month_days:
            row: Dict[str, Any] = {"date": d.isoformat()}

            if d > last_avail:
                row["mean_mm"] = None
                row["status"] = "pending"
                out_series.append(row)
                continue

            ti = date_to_idx.get(d)
            if ti is None:
                row["mean_mm"] = None
                row["status"] = "missing"
                out_series.append(row)
                continue

            arr = vpr[ti, ys, xs]  # netCDF4 masked array typically
            if np.ma.isMaskedArray(arr):
                data = np.array(np.ma.filled(arr, np.nan), dtype=np.float32)
            else:
                data = np.array(arr, dtype=np.float32)

            vals = data[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row["mean_mm"] = None
                row["status"] = "nodata"
            else:
                m = float(np.mean(vals))
                row["mean_mm"] = round(m, 1)
                row["status"] = "ok"
            out_series.append(row)

    ok_vals = [r["mean_mm"] for r in out_series if r.get("status") == "ok" and isinstance(r.get("mean_mm"), (int, float))]
    ok_vals_f = np.array(ok_vals, dtype=np.float32) if ok_vals else np.array([], dtype=np.float32)

    def _safe(x):
        return float(x) if np.isfinite(x) else None

    stats = {
        "count_ok": int(ok_vals_f.size),
        "sum_mm": _safe(np.sum(ok_vals_f)) if ok_vals_f.size else None,
        "mean_mm": _safe(np.mean(ok_vals_f)) if ok_vals_f.size else None,
        "median_mm": _safe(np.median(ok_vals_f)) if ok_vals_f.size else None,
        "max_mm": _safe(np.max(ok_vals_f)) if ok_vals_f.size else None,
        "p90_mm": _safe(np.percentile(ok_vals_f, 90)) if ok_vals_f.size else None,
        "dry_days_lt_1mm": int(np.sum(ok_vals_f < 1.0)) if ok_vals_f.size else 0,
        "wet_days_ge_10mm": int(np.sum(ok_vals_f >= 10.0)) if ok_vals_f.size else 0,
        "wet_days_ge_20mm": int(np.sum(ok_vals_f >= 20.0)) if ok_vals_f.size else 0,
        "aoi_area_km2": None,  # filled in caller
        "variable": "HYRAS daily precipitation (mean over AOI)",
    }

    return out_series, stats, last_avail.isoformat()


# -------------------------------
# Plotting
# -------------------------------

def _plot_calendar(year: int, month: int, series: List[Dict[str, Any]], out_png: Path) -> None:
    m = {r["date"]: r for r in series}
    cal = calendar.Calendar(firstweekday=0)  # Monday
    weeks = cal.monthdayscalendar(year, month)

    vals = [r.get("mean_mm") for r in series if r.get("status") == "ok" and isinstance(r.get("mean_mm"), (int, float))]
    vmax = float(np.percentile(np.array(vals, dtype=np.float32), 95)) if vals else 10.0
    vmax = max(1.0, vmax)

    fig = plt.figure(figsize=(12, 7), dpi=140)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    title = f"Tagesniederschlag (AOI-Mittel) – {_month_name_de(month)} {year}"
    ax.text(0.0, 1.02, title, fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.0, 0.98,
        "Einheit: mm/Tag · Werte = Flächenmittel im Polygon · HYRAS (DWD CDC) · Tage ohne Daten: n/a",
        fontsize=10, alpha=0.75, transform=ax.transAxes
    )

    n_rows = len(weeks)
    n_cols = 7
    x0, y0 = 0.02, 0.05
    grid_w, grid_h = 0.96, 0.86
    cell_w = grid_w / n_cols
    cell_h = grid_h / n_rows

    weekdays = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    for c, wd in enumerate(weekdays):
        ax.text(x0 + c * cell_w + 0.005, y0 + grid_h + 0.01, wd, fontsize=11, alpha=0.85, transform=ax.transAxes)

    cmap = plt.cm.Blues

    for r in range(n_rows):
        for c in range(n_cols):
            daynum = weeks[r][c]
            cx = x0 + c * cell_w
            cy = y0 + (n_rows - 1 - r) * cell_h

            face = (1, 1, 1, 0.04)
            edge = (1, 1, 1, 0.10)

            label = ""
            sub = ""

            if daynum == 0:
                face = (1, 1, 1, 0.02)
                edge = (1, 1, 1, 0.06)
            else:
                d = date(year, month, daynum).isoformat()
                row = m.get(d, {})
                status = row.get("status")
                v = row.get("mean_mm")

                label = str(daynum)

                if status == "ok" and isinstance(v, (int, float)):
                    t = min(1.0, max(0.0, float(v) / vmax))
                    face = cmap(0.15 + 0.75 * t)
                    sub = f"{float(v):.1f} mm"
                elif status in ("pending", "missing", "nodata"):
                    face = (1, 1, 1, 0.03)
                    sub = "n/a"

            ax.add_patch(Rectangle((cx, cy), cell_w, cell_h, transform=ax.transAxes,
                                   facecolor=face, edgecolor=edge, linewidth=1.0))

            if daynum != 0:
                ax.text(cx + 0.008, cy + cell_h - 0.035, label, fontsize=11, fontweight="bold",
                        transform=ax.transAxes, alpha=0.90)
                if sub:
                    ax.text(cx + 0.008, cy + 0.020, sub, fontsize=12,
                            transform=ax.transAxes, alpha=0.92)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_bars(year: int, month: int, series: List[Dict[str, Any]], out_png: Path) -> None:
    xs = []
    ys = []
    for r in series:
        if r.get("status") == "ok" and isinstance(r.get("mean_mm"), (int, float)):
            d = datetime.fromisoformat(r["date"]).date()
            xs.append(d.day)
            ys.append(float(r["mean_mm"]))

    fig = plt.figure(figsize=(12, 4.2), dpi=140)
    ax = fig.add_subplot(111)
    ax.set_xticks(xs)
    ax.set_title(f"Tageswerte (AOI-Mittel) – {_month_name_de(month)} {year}", fontsize=14, fontweight="bold")

    if ys:
        ax.bar(xs, ys)
        ax.set_xlabel("Tag")
        ax.set_ylabel("mm/Tag")
        ax.set_xlim(0.5, calendar.monthrange(year, month)[1] + 0.5)
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Keine Werte verfügbar.", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative(year: int, month: int, series: List[Dict[str, Any]], out_png: Path) -> None:
    days = []
    cum = []
    s = 0.0
    for r in series:
        if r.get("status") == "ok" and isinstance(r.get("mean_mm"), (int, float)):
            d = datetime.fromisoformat(r["date"]).date()
            s += float(r["mean_mm"])
            days.append(d.day)
            cum.append(s)

    fig = plt.figure(figsize=(12, 4.2), dpi=140)
    ax = fig.add_subplot(111)
    ax.set_xticks(days)
    ax.set_title(f"Kumulierte Summe (AOI-Mittel) – {_month_name_de(month)} {year}", fontsize=14, fontweight="bold")

    if cum:
        ax.plot(days, cum, marker="o")
        ax.set_xlabel("Tag")
        ax.set_ylabel("kumuliert (mm)")
        ax.set_xlim(0.5, calendar.monthrange(year, month)[1] + 0.5)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Keine Werte verfügbar.", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _write_csv(series: List[Dict[str, Any]], out_csv: Path) -> None:
    lines = ["date,mean_mm,status"]
    for r in series:
        v = r.get("mean_mm")
        vstr = "" if v is None else str(v)
        lines.append(f"{r.get('date','')},{vstr},{r.get('status','')}")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------
# Range (1/3/6/12 months or explicit)
# -------------------------------

def _parse_iso_date(s: Any) -> date:
    if not s:
        raise ValueError("Datum fehlt (YYYY-MM-DD).")
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s.strip()).date()
        except Exception:
            pass
    raise ValueError(f"Ungültiges Datum: {s!r} (erwartet YYYY-MM-DD).")


def _month_shift(year: int, month: int, delta_months: int) -> Tuple[int, int]:
    m = month + delta_months
    y = year
    while m <= 0:
        y -= 1
        m += 12
    while m > 12:
        y += 1
        m -= 12
    return y, m


def _period_months_to_dates(months: int, end_d: date) -> Tuple[date, date]:
    if months not in (1, 3, 6, 12):
        raise ValueError("months muss 1, 3, 6 oder 12 sein.")
    sy, sm = _month_shift(end_d.year, end_d.month, -(months - 1))
    start = date(sy, sm, 1)
    return start, end_d


def _iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


@lru_cache(maxsize=8)
def _hyras_last_available_date_for_year(year: int) -> date:
    _cleanup_cache()
    fn = _find_hyras_daily_file(year)
    url = HYRAS_DAILY_BASE.rstrip("/") + "/" + fn
    local_nc = TMP_DIR / fn
    _http_download_file(url, local_nc, tries=3)

    with Dataset(str(local_nc), mode="r") as nc:
        tvar = None
        for cand in ("time", "TIME", "t"):
            if cand in nc.variables:
                tvar = nc.variables[cand]
                break
        if tvar is None:
            for name, vv in nc.variables.items():
                if getattr(vv, "ndim", 0) == 1:
                    u = (getattr(vv, "units", "") or "").lower()
                    if "since" in u and "day" in u:
                        tvar = vv
                        break
        if tvar is None:
            raise ValueError("Keine Zeitvariable im NetCDF gefunden (für last_available_date).")

        t_units = getattr(tvar, "units", None)
        t_cal = getattr(tvar, "calendar", "standard")
        tvals = np.array(tvar[:])
        dts = num2date(tvals, units=t_units, calendar=t_cal)
        dates = [date(dt.year, dt.month, dt.day) for dt in dts]
        if not dates:
            raise ValueError(f"Keine Datumswerte im Jahr {year}.")
        return max(dates)


def _compute_range_series_hyras(
    gj: Dict[str, Any],
    start_d: date,
    end_d: date,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, float]:
    _cleanup_cache()

    if end_d < start_d:
        raise ValueError("end_date muss >= start_date sein.")
    n_days = (end_d - start_d).days + 1
    if n_days > MAX_RANGE_DAYS:
        raise ValueError(f"Zeitraum zu groß: {n_days} Tage (Limit: {MAX_RANGE_DAYS}).")

    geom_wgs84 = _extract_single_geometry(gj)
    if geom_wgs84.is_empty:
        raise ValueError("Geometrie ist leer.")
    if geom_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Nur Polygon/MultiPolygon erlaubt (bekommen: {geom_wgs84.geom_type}).")

    geom_3035 = _geom_to_epsg(geom_wgs84, 4326, HYRAS_EPSG)
    aoi_area_km2 = float(geom_3035.area) / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    years = list(range(start_d.year, end_d.year + 1))

    ys = xs = None
    mask = None
    last_avail_end_year = None

    out_series: List[Dict[str, Any]] = []

    for yr in years:
        fn = _find_hyras_daily_file(yr)
        url = HYRAS_DAILY_BASE.rstrip("/") + "/" + fn
        local_nc = TMP_DIR / fn
        _http_download_file(url, local_nc, tries=3)

        with Dataset(str(local_nc), mode="r") as nc:
            _, vpr = _pick_precip_variable(nc)

            tvar = None
            for cand in ("time", "TIME", "t"):
                if cand in nc.variables:
                    tvar = nc.variables[cand]
                    break
            if tvar is None:
                for name, vv in nc.variables.items():
                    if getattr(vv, "ndim", 0) == 1:
                        u = (getattr(vv, "units", "") or "").lower()
                        if "since" in u and "day" in u:
                            tvar = vv
                            break
            if tvar is None:
                raise ValueError("Keine Zeitvariable im NetCDF gefunden.")

            t_units = getattr(tvar, "units", None)
            t_cal = getattr(tvar, "calendar", "standard")
            tvals = np.array(tvar[:])
            dts = num2date(tvals, units=t_units, calendar=t_cal)
            dates = [date(dt.year, dt.month, dt.day) for dt in dts]
            date_to_idx = {d: i for i, d in enumerate(dates)}
            last_avail = max(dates) if dates else date(yr, 12, 31)

            if yr == end_d.year:
                last_avail_end_year = last_avail

            if ys is None or xs is None or mask is None:
                minx, miny, maxx, maxy = geom_3035.bounds
                x, y = _get_xy(nc)
                ys2, xs2 = _subset_slices_from_bounds(x, y, (minx, miny, maxx, maxy))

                y_sub = y[ys2]
                x_sub = x[xs2]
                h = int(y_sub.size)
                w = int(x_sub.size)
                if h <= 0 or w <= 0:
                    raise ValueError("Subset leer (h/w <= 0).")
                if (h * w) > MAX_SUBSET_PIXELS:
                    raise ValueError(f"AOI-Subset zu groß: {h*w} Pixel (Limit: {MAX_SUBSET_PIXELS}).")

                transform = _affine_for_subset(x, y, ys2, xs2)
                mask2 = _aoi_mask_for_subset(geom_3035, (h, w), transform)
                if int(np.sum(mask2)) <= 0:
                    raise ValueError("AOI-Maske leer (Polygon schneidet Rasterzellen nicht).")

                ys, xs, mask = ys2, xs2, mask2

            y_start = max(start_d, date(yr, 1, 1))
            y_end = min(end_d, date(yr, 12, 31))
            year_dates = list(_iter_dates(y_start, y_end))

            ti_list: List[int] = []
            positions: List[int] = []

            for d in year_dates:
                row = {"date": d.isoformat()}

                if d > last_avail:
                    row["mean_mm"] = None
                    row["status"] = "pending"
                else:
                    ti = date_to_idx.get(d)
                    if ti is None:
                        row["mean_mm"] = None
                        row["status"] = "missing"
                    else:
                        row["mean_mm"] = None
                        row["status"] = "calc"
                        ti_list.append(int(ti))
                        positions.append(len(out_series))

                out_series.append(row)

            if ti_list:
                arr3 = vpr[np.array(ti_list, dtype=np.int64), ys, xs]
                if np.ma.isMaskedArray(arr3):
                    arr3 = np.ma.filled(arr3, np.nan)
                data = np.array(arr3, dtype=np.float32)

                vals = data[:, mask]
                means = np.nanmean(vals, axis=1)

                for pos, mm in zip(positions, means.tolist()):
                    if mm is None or (isinstance(mm, float) and not np.isfinite(mm)):
                        out_series[pos]["mean_mm"] = None
                        out_series[pos]["status"] = "nodata"
                    else:
                        out_series[pos]["mean_mm"] = round(float(mm), 1)
                        out_series[pos]["status"] = "ok"

    if last_avail_end_year is None:
        last_avail_end_year = end_d

    last_avail_iso = last_avail_end_year.isoformat()

    ok_vals = [r["mean_mm"] for r in out_series if r.get("status") == "ok" and isinstance(r.get("mean_mm"), (int, float))]
    ok_vals_f = np.array(ok_vals, dtype=np.float32) if ok_vals else np.array([], dtype=np.float32)

    def _safe(x):
        return float(x) if np.isfinite(x) else None

    stats = {
        "count_ok": int(ok_vals_f.size),
        "sum_mm": _safe(np.sum(ok_vals_f)) if ok_vals_f.size else None,
        "mean_mm": _safe(np.mean(ok_vals_f)) if ok_vals_f.size else None,
        "median_mm": _safe(np.median(ok_vals_f)) if ok_vals_f.size else None,
        "max_mm": _safe(np.max(ok_vals_f)) if ok_vals_f.size else None,
        "p90_mm": _safe(np.percentile(ok_vals_f, 90)) if ok_vals_f.size else None,
        "dry_days_lt_1mm": int(np.sum(ok_vals_f < 1.0)) if ok_vals_f.size else 0,
        "wet_days_ge_10mm": int(np.sum(ok_vals_f >= 10.0)) if ok_vals_f.size else 0,
        "wet_days_ge_20mm": int(np.sum(ok_vals_f >= 20.0)) if ok_vals_f.size else 0,
        "variable": "HYRAS daily precipitation (mean over AOI)",
    }

    return out_series, stats, last_avail_iso, aoi_area_km2


def _compute_range(gj: Dict[str, Any], start_d: date, end_d: date) -> Tuple[str, Path, Path, Dict[str, Any]]:
    series, stats, last_avail_iso, aoi_area_km2 = _compute_range_series_hyras(gj, start_d, end_d)

    job_id = uuid.uuid4().hex[:12]
    csv_path = TMP_DIR / f"{job_id}.range.csv"
    json_path = TMP_DIR / f"{job_id}.range.json"

    _write_csv(series, csv_path)

    payload = {
        "job_id": job_id,
        "mode": "range",
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "last_available_date": last_avail_iso,
        "aoi_area_km2": round(aoi_area_km2, 4),
        "series": series,
        "stats": {**stats, "aoi_area_km2": round(aoi_area_km2, 4)},
        "source": {
            "dataset": "HYRAS-DE-PR daily (DWD CDC)",
            "crs": "EPSG:3035",
            "base_url": HYRAS_DAILY_BASE,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return job_id, csv_path, json_path, payload


# -------------------------------
# Compute pipeline (month report)
# -------------------------------

def _compute(gj: Dict[str, Any], month: int) -> ComputeResult:
    _cleanup_cache()

    now = datetime.now(timezone.utc)
    year = now.year

    if month < 1 or month > 12:
        raise ValueError("Monat muss 1..12 sein.")
    if month > now.month:
        raise ValueError(f"Monat {month} liegt in der Zukunft (aktueller Monat: {now.month}).")

    geom_wgs84 = _extract_single_geometry(gj)
    geom_3035 = _geom_to_epsg(geom_wgs84, 4326, HYRAS_EPSG)
    aoi_area_km2 = float(geom_3035.area) / 1_000_000.0
    if MAX_AOI_AREA_KM2 > 0 and aoi_area_km2 > MAX_AOI_AREA_KM2:
        raise ValueError(f"AOI ist zu groß: {aoi_area_km2:.3f} km² (Limit: {MAX_AOI_AREA_KM2:.3f} km²).")

    series, stats, last_avail = _compute_month_series_hyras(gj, year, month)
    stats["aoi_area_km2"] = round(aoi_area_km2, 4)

    job_id = uuid.uuid4().hex[:12]

    cal_png = TMP_DIR / f"{job_id}.calendar.png"
    bar_png = TMP_DIR / f"{job_id}.bars.png"
    cum_png = TMP_DIR / f"{job_id}.cumulative.png"
    csv_path = TMP_DIR / f"{job_id}.daily.csv"
    json_path = TMP_DIR / f"{job_id}.result.json"

    _plot_calendar(year, month, series, cal_png)
    _plot_bars(year, month, series, bar_png)
    _plot_cumulative(year, month, series, cum_png)
    _write_csv(series, csv_path)

    payload = {
        "job_id": job_id,
        "year": year,
        "month": month,
        "month_name": _month_name_de(month),
        "aoi_area_km2": aoi_area_km2,
        "last_available_date": last_avail,
        "series": series,
        "stats": stats,
        "source": {
            "dataset": "HYRAS-DE-PR daily (DWD CDC)",
            "crs": "EPSG:3035",
            "base_url": HYRAS_DAILY_BASE,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ComputeResult(
        job_id=job_id,
        year=year,
        month=month,
        month_name=_month_name_de(month),
        aoi_area_km2=aoi_area_km2,
        last_available_date=last_avail,
        series=series,
        stats=stats,
        calendar_png=cal_png,
        bars_png=bar_png,
        cumulative_png=cum_png,
        csv_path=csv_path,
        json_path=json_path,
    )


# -------------------------------
# Routes / UI
# -------------------------------

INDEX_HTML = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />

  <style>
    :root{
      --bg:#0b0f19;
      --card:#111a2e;
      --text:#e6eaf2;
      --muted:#a8b3cf;
      --border: rgba(255,255,255,.10);
      --primary:#6ea8fe;
      --focus: rgba(110,168,254,.45);
      --radius: 16px;
      --container: 1200px;
      --gap: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    body{ margin:0; font-family: var(--font); background: var(--bg); color: var(--text); }
    .wrap{ max-width: var(--container); margin: 18px auto; padding: 0 14px 24px; display: grid; grid-template-columns: 1.2fr .8fr; gap: var(--gap); }
    header{ max-width: var(--container); margin: 18px auto 0; padding: 0 14px; display:flex; align-items:baseline; justify-content:space-between; gap: 12px; }
    h1{ font-size: 18px; margin:0; letter-spacing: .2px; }
    .hint{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.35; }
    .card{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: 0 18px 60px rgba(0,0,0,.35); overflow: hidden; }
    #map{ height: 70vh; min-height: 520px; }
    .panel{ padding: 12px; display:flex; flex-direction:column; gap: 10px; }
    label{ color: var(--muted); font-size: 12px; }
    textarea{
      width: 100%; min-height: 160px; resize: vertical;
      background: rgba(255,255,255,.04); border: 1px solid var(--border); border-radius: 12px;
      padding: 10px; color: var(--text); font-family: var(--mono); font-size: 12px; outline: none;
    }
    textarea:focus{ border-color: var(--primary); box-shadow: 0 0 0 4px var(--focus); }
    .row{ display:flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button{
      appearance:none; border: 1px solid var(--border); background: rgba(255,255,255,.06);
      color: var(--text); padding: 10px 12px; border-radius: 12px; cursor: pointer; font-weight: 600;
    }
    button.primary{ border-color: rgba(110,168,254,.35); background: rgba(110,168,254,.16); }
    button:disabled{ opacity:.55; cursor:not-allowed; }
    select,input{
      background: rgba(255,255,255,.04); border: 1px solid var(--border); border-radius: 10px;
      padding: 8px 10px; color: var(--text);
    }
    .status{
      color: var(--muted); font-size: 13px; line-height: 1.35;
      padding: 8px 10px; border-radius: 12px; background: rgba(0,0,0,.18); border: 1px solid var(--border);
    }
    .status b{ color: var(--text); }
    .err{ border-color: rgba(255,100,100,.35); background: rgba(255,100,100,.10); color: #ffd1d1; }
    .ok{ border-color: rgba(120,220,160,.35); background: rgba(120,220,160,.08); }
    .small{ font-size: 12px; color: var(--muted); }
    .mono{ font-family: var(--mono); }
    img{ max-width:100%; border-radius: 12px; border: 1px solid rgba(255,255,255,.10); }
    table{ width:100%; border-collapse: collapse; font-size: 12px; }
    th,td{ padding: 6px 6px; border-bottom: 1px solid rgba(255,255,255,.08); vertical-align: top; }
    th{ color: var(--muted); font-weight: 600; text-align:left; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ title }}</h1>
      <div class="hint">
        Zeichne ein Polygon/Rechteck (immer nur <b>ein</b> Feature). Wähle den Monat im aktuellen Jahr und erstelle:
        <b>Kalenderplot</b> (mm/Tag, AOI-Mittel) + Balken + kumuliert + Statistik.
      </div>
    </div>
    <div class="small">API: <code>/api/compute</code> · <code>/api/range</code> · Downloads unter <code>/r/&lt;job&gt;/...</code></div>
  </header>

  <div class="wrap">
    <div class="card"><div id="map"></div></div>

    <div class="card">
      <div class="panel">
        <div class="row">
          <button id="btn-clear">AOI löschen</button>
        </div>

        <div class="row" style="margin-top:6px;">
          <label>Zeitraum-CSV:
            <select id="rangeMonths">
              <option value="1">1 Monat</option>
              <option value="3">3 Monate</option>
              <option value="6">6 Monate</option>
              <option value="12">12 Monate</option>
            </select>
          </label>
          <button id="btn-range-csv" disabled>CSV für Zeitraum erstellen</button>
        </div>
        <div class="small">
          Exportiert tägliche AOI-Mittelwerte als CSV für die letzten N Monate (Monatsgrenzen), bis zum letzten verfügbaren HYRAS-Tag.
        </div>


        <div class="row">
          <label>Monat:
            <select id="month">
              {% for m in months %}
                <option value="{{ m.value }}" {% if m.value == default_month %}selected{% endif %}>{{ m.label }}</option>
              {% endfor %}
            </select>
          </label>
            <button class="primary" id="btn-run" disabled>Monatsreport erstellen</button>

          <div class="small">Jahr: <b>{{ year }}</b> · Limit AOI ≤ <b>{{ max_area_km2 }} km²</b></div>
        </div>

        <div id="status" class="status">Noch keine AOI.</div>



        <label>GeoJSON (aktuelles Feature, EPSG:4326)</label>
        <textarea id="geojson" spellcheck="false" placeholder="Hier erscheint das GeoJSON…"></textarea>

        <div id="outBox" style="display:none;">
          <div class="small">Monats-Statistik:</div>
          <div style="overflow:auto; margin-top:6px;">
            <table>
              <tbody id="statsBody"></tbody>
            </table>
          </div>

          <div class="row" style="margin-top:10px;">
            <button id="btn-dl-csv" disabled>CSV herunterladen</button>
            <button id="btn-dl-json" disabled>JSON herunterladen</button>
            <button id="btn-dl-cal" disabled>Kalender PNG</button>
            <button id="btn-dl-bars" disabled>Balken PNG</button>
            <button id="btn-dl-cum" disabled>Kumuliert PNG</button>
          </div>

          <div style="margin-top:10px; display:flex; flex-direction:column; gap:10px;">
            <img id="imgCal" alt="Kalenderplot"/>
            <img id="imgBars" alt="Tagesbalken"/>
            <img id="imgCum" alt="Kumuliert"/>
          </div>
        </div>

        <div class="small" style="margin-top:10px;">
          Quelle: DWD CDC HYRAS-DE-PR daily (NetCDF, EPSG:3035). Tagesdefinition: 6UTC–6UTC (wie in der Datensatzbeschreibung).
        </div>
      </div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <script>
    const map = L.map('map', { preferCanvas: true }).setView([49.87, 8.65], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 20, attribution: '&copy; OpenStreetMap' }).addTo(map);

    const drawn = new L.FeatureGroup().addTo(map);
    const drawControl = new L.Control.Draw({
      draw: { polyline:false, circle:false, circlemarker:false, marker:false, polygon:{ allowIntersection:false, showArea:true }, rectangle:true },
      edit: { featureGroup: drawn, edit:true, remove:true }
    });
    map.addControl(drawControl);

    let currentFeature = null;

    const elGeo = document.getElementById('geojson');
    const elStatus = document.getElementById('status');
    const btnClear = document.getElementById('btn-clear');
    const btnRun = document.getElementById('btn-run');
    const elMonth = document.getElementById('month');

    const elRangeMonths = document.getElementById('rangeMonths');
    const btnRangeCSV = document.getElementById('btn-range-csv');

    const outBox = document.getElementById('outBox');
    const statsBody = document.getElementById('statsBody');

    const imgCal = document.getElementById('imgCal');
    const imgBars = document.getElementById('imgBars');
    const imgCum = document.getElementById('imgCum');

    const btnDlCSV = document.getElementById('btn-dl-csv');
    const btnDlJSON = document.getElementById('btn-dl-json');
    const btnDlCal = document.getElementById('btn-dl-cal');
    const btnDlBars = document.getElementById('btn-dl-bars');
    const btnDlCum = document.getElementById('btn-dl-cum');

    function setStatus(html, cls){
      elStatus.className = 'status' + (cls ? (' ' + cls) : '');
      elStatus.innerHTML = html;
    }

    function featureToGeoJSON(layer){
      return { type: "Feature", properties: { epsg: 4326 }, geometry: layer.toGeoJSON().geometry };
    }

    function setButtons(){
      btnRun.disabled = !currentFeature;
      btnRangeCSV.disabled = !currentFeature;
    }

    function clearAll(){
      drawn.clearLayers();
      currentFeature = null;
      elGeo.value = '';
      outBox.style.display = 'none';
      statsBody.innerHTML = '';
      btnDlCSV.disabled = true;
      btnDlJSON.disabled = true;
      btnDlCal.disabled = true;
      btnDlBars.disabled = true;
      btnDlCum.disabled = true;
      btnRangeCSV.disabled = true;
      setButtons();
      setStatus('Noch keine AOI.', '');
    }

    map.on(L.Draw.Event.CREATED, function (e) {
      drawn.clearLayers();
      const layer = e.layer;
      drawn.addLayer(layer);
      currentFeature = layer;

      const gj = featureToGeoJSON(layer);
      elGeo.value = JSON.stringify(gj, null, 2);

      outBox.style.display = 'none';
      statsBody.innerHTML = '';
      setButtons();
      setStatus('AOI gesetzt. Jetzt <b>Monatsreport berechnen</b>.', 'ok');
    });

    map.on('draw:edited', function(){
      const layers = drawn.getLayers();
      if(layers.length < 1) return;
      currentFeature = layers[0];
      const gj = featureToGeoJSON(currentFeature);
      elGeo.value = JSON.stringify(gj, null, 2);

      outBox.style.display = 'none';
      statsBody.innerHTML = '';
      setButtons();
      setStatus('AOI geändert. Bitte <b>Monatsreport berechnen</b> erneut ausführen.', 'ok');
    });

    map.on('draw:deleted', function(){
      clearAll();
    });

    btnClear.addEventListener('click', clearAll);

    async function apiJson(url, body){
      const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const ct = (res.headers.get('content-type')||'').toLowerCase();
      const raw = await res.text();
      if(!ct.includes('application/json') && !ct.includes('json')){
        throw new Error(`Server lieferte kein JSON (HTTP ${res.status}, Content-Type=${ct}). Antwort-Auszug: ${raw.slice(0,240)}`);
      }
      const js = raw ? JSON.parse(raw) : {};
      if(!res.ok){
        throw new Error(js && js.error ? js.error : (`HTTP ${res.status}`));
      }
      return js;
    }

    function fillStats(stats, meta){
      statsBody.innerHTML = '';
      const rows = [
        ["AOI Fläche (km²)", (meta.aoi_area_km2 ?? '').toString()],
        ["Letztes verfügbares Datum", (meta.last_available_date ?? '').toString()],
        ["OK-Tage", (stats.count_ok ?? '').toString()],
        ["Summe (mm)", (stats.sum_mm ?? '').toString()],
        ["Mittel (mm)", (stats.mean_mm ?? '').toString()],
        ["Median (mm)", (stats.median_mm ?? '').toString()],
        ["Max (mm)", (stats.max_mm ?? '').toString()],
        ["p90 (mm)", (stats.p90_mm ?? '').toString()],
        ["Trockentage < 1mm", (stats.dry_days_lt_1mm ?? '').toString()],
        ["Starktregentage ≥ 10mm", (stats.wet_days_ge_10mm ?? '').toString()],
        ["Sehr stark ≥ 20mm", (stats.wet_days_ge_20mm ?? '').toString()],
      ];
      for(const [k,v] of rows){
        const tr = document.createElement('tr');
        tr.innerHTML = `<th>${k}</th><td class="mono">${v}</td>`;
        statsBody.appendChild(tr);
      }
    }

    btnRun.addEventListener('click', async () => {
      try{
        if(!currentFeature) return;

        let gj;
        try{ gj = JSON.parse(elGeo.value); }catch(e){ setStatus('GeoJSON ist ungültig.', 'err'); return; }

        const month = Number(elMonth.value || 1);
        setStatus('Berechne Monatsreport… (Download/Subset kann je nach Monat/AOI etwas dauern)', '');

        const data = await apiJson('/api/compute', { geojson: gj, month });

        outBox.style.display = 'block';
        fillStats(data.stats || {}, data.meta || {});

        imgCal.src = data.download.calendar_png;
        imgBars.src = data.download.bars_png;
        imgCum.src = data.download.cumulative_png;

        btnDlCSV.disabled = false;
        btnDlJSON.disabled = false;
        btnDlCal.disabled = false;
        btnDlBars.disabled = false;
        btnDlCum.disabled = false;

        btnDlCSV.onclick = () => window.location = data.download.csv;
        btnDlJSON.onclick = () => window.location = data.download.json;
        btnDlCal.onclick = () => window.location = data.download.calendar_png;
        btnDlBars.onclick = () => window.location = data.download.bars_png;
        btnDlCum.onclick = () => window.location = data.download.cumulative_png;

        setStatus(`Fertig. Monat: <b>${data.meta.month_name} ${data.meta.year}</b> · Job: <span class="mono">${data.job_id}</span>`, 'ok');
      }catch(e){
        setStatus('Fehler: ' + (e && e.message ? e.message : String(e)), 'err');
      }
    });

    btnRangeCSV.addEventListener('click', async () => {
      try{
        if(!currentFeature) return;

        let gj;
        try{ gj = JSON.parse(elGeo.value); }
        catch(e){ setStatus('GeoJSON ist ungültig.', 'err'); return; }

        const months = Number(elRangeMonths.value || 1);
        setStatus(`Erstelle Zeitraum-CSV (${months} Monat(e))…`, '');

        const data = await apiJson('/api/range', { geojson: gj, months });

        const meta = data.meta || {};
        setStatus(
          `CSV bereit: <b>${meta.start_date}</b> bis <b>${meta.end_date}</b> · Job: <span class="mono">${data.job_id}</span>`,
          'ok'
        );

        window.location = data.download.csv;
      }catch(e){
        setStatus('Fehler: ' + (e && e.message ? e.message : String(e)), 'err');
      }
    });

    clearAll();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    now = datetime.now(timezone.utc)
    year = now.year
    cur_m = now.month
    dm = DEFAULT_MONTH if 1 <= DEFAULT_MONTH <= 12 else cur_m

    months = [{"value": m, "label": f"{m:02d} – {_month_name_de(m)}"} for m in range(1, 13)]
    return render_template_string(
        INDEX_HTML,
        title=APP_TITLE,
        year=year,
        months=months,
        default_month=dm,
        max_area_km2=MAX_AOI_AREA_KM2,
    )


@app.route("/api/compute", methods=["POST", "OPTIONS"])
def api_compute():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))
        month = int(body.get("month", 0) or 0)
        if month <= 0:
            month = datetime.now(timezone.utc).month

        rr = _compute(gj, month)

        return jsonify({
            "ok": True,
            "job_id": rr.job_id,
            "stats": rr.stats,
            "meta": {
                "year": rr.year,
                "month": rr.month,
                "month_name": rr.month_name,
                "aoi_area_km2": round(rr.aoi_area_km2, 4),
                "last_available_date": rr.last_available_date,
            },
            "download": {
                "calendar_png": f"/r/{rr.job_id}/calendar.png",
                "bars_png": f"/r/{rr.job_id}/bars.png",
                "cumulative_png": f"/r/{rr.job_id}/cumulative.png",
                "csv": f"/r/{rr.job_id}/daily.csv",
                "json": f"/r/{rr.job_id}/result.json",
            },
        })
    except Exception as e:
        return _json_error(str(e), 400)


@app.route("/api/range", methods=["POST", "OPTIONS"])
def api_range():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        body = request.get_json(force=True, silent=False) or {}
        gj = _parse_geojson(body.get("geojson"))

        months = body.get("months", None)
        start_s = body.get("start_date", None)
        end_s = body.get("end_date", None)

        if start_s or end_s:
            if not start_s or not end_s:
                raise ValueError("Wenn start_date angegeben wird, muss auch end_date angegeben werden (und umgekehrt).")
            start_d = _parse_iso_date(start_s)
            end_d = _parse_iso_date(end_s)
        else:
            m = int(months or 0)
            if m not in (1, 3, 6, 12):
                raise ValueError("months muss 1, 3, 6 oder 12 sein (oder nutze start_date/end_date).")

            now_d = datetime.now(timezone.utc).date()
            last_avail = _hyras_last_available_date_for_year(now_d.year)
            end_d = min(now_d, last_avail)

            # edge case: if now_d is in a new year but last_avail isn't yet updated and is < end_d
            if last_avail < end_d:
                end_d = last_avail

            start_d, end_d = _period_months_to_dates(m, end_d)

        job_id, _, _, payload = _compute_range(gj, start_d, end_d)

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "stats": payload.get("stats", {}),
            "meta": {
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "last_available_date": payload.get("last_available_date"),
                "aoi_area_km2": payload.get("aoi_area_km2"),
            },
            "download": {
                "csv": f"/r/{job_id}/range.csv",
                "json": f"/r/{job_id}/range.json",
            },
        })
    except Exception as e:
        return _json_error(str(e), 400)


@app.get("/r/<job_id>/calendar.png")
def dl_calendar(job_id: str):
    p = TMP_DIR / f"{job_id}.calendar.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=True, download_name=f"precip_calendar_{job_id}.png", conditional=True)


@app.get("/r/<job_id>/bars.png")
def dl_bars(job_id: str):
    p = TMP_DIR / f"{job_id}.bars.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=True, download_name=f"precip_bars_{job_id}.png", conditional=True)


@app.get("/r/<job_id>/cumulative.png")
def dl_cumulative(job_id: str):
    p = TMP_DIR / f"{job_id}.cumulative.png"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="image/png", as_attachment=True, download_name=f"precip_cumulative_{job_id}.png", conditional=True)


@app.get("/r/<job_id>/daily.csv")
def dl_csv(job_id: str):
    p = TMP_DIR / f"{job_id}.daily.csv"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="text/csv", as_attachment=True, download_name=f"precip_daily_{job_id}.csv", conditional=True)


@app.get("/r/<job_id>/result.json")
def dl_json(job_id: str):
    p = TMP_DIR / f"{job_id}.result.json"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="application/json", as_attachment=True, download_name=f"precip_result_{job_id}.json", conditional=True)


@app.get("/r/<job_id>/range.csv")
def dl_range_csv(job_id: str):
    p = TMP_DIR / f"{job_id}.range.csv"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="text/csv", as_attachment=True, download_name=f"precip_range_{job_id}.csv", conditional=True)


@app.get("/r/<job_id>/range.json")
def dl_range_json(job_id: str):
    p = TMP_DIR / f"{job_id}.range.json"
    if not p.exists():
        return jsonify({"error": "Job nicht gefunden/abgelaufen."}), 404
    return send_file(p, mimetype="application/json", as_attachment=True, download_name=f"precip_range_{job_id}.json", conditional=True)


@app.get("/healthz")
def healthz():
    return Response("ok", mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
