#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic stations functions for seismo-rain-analytics project.
"""

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def _parse_offsets(offsets_str: str | None):
    if not offsets_str:
        # dlon, dlat (grados)
        return [(0.25, 0.20), (-0.35, 0.15), (0.10, -0.30), (-0.20, -0.25)]
    pairs = []
    for part in offsets_str.split(';'):
        part = part.strip()
        if not part:
            continue
        x, y = part.split(',')
        pairs.append((float(x.strip()), float(y.strip())))
    return pairs

def _parse_values(values_str: str | None):
    if not values_str:
        return [5.0, 12.0, 25.0, 40.0]
    return [float(x.strip()) for x in values_str.split(',') if x.strip()]

def add_synthetic(df: pd.DataFrame, base_name: str = "REVS2",
                  offsets_str: str | None = None,
                  values_str: str | None = None) -> pd.DataFrame:
    """
    Agrega estaciones sintéticas alrededor de una estación base (por defecto REVS2).
    - offsets: lista de (dlon, dlat) en grados
    - valores: lista de lluvias (mm)
    """
    if df.empty:
        return df
    offsets = _parse_offsets(offsets_str)
    values = _parse_values(values_str)
    k = min(len(offsets), len(values))
    if k == 0:
        return df

    # Punto base
    base = df.loc[df["name"] == base_name]
    if not base.empty:
        base_lon = float(base.iloc[0]["lon"])
        base_lat = float(base.iloc[0]["lat"])
    else:
        base_lon = float(df["lon"].mean())
        base_lat = float(df["lat"].mean())

    syn_rows = []
    idx = 1
    existing = set(df["name"].tolist())
    for (dlon, dlat), v in zip(offsets[:k], values[:k]):
        name = f"SYN{idx}"
        while name in existing:
            idx += 1
            name = f"SYN{idx}"
        syn_rows.append({
            "name": name,
            "lat": base_lat + dlat,
            "lon": base_lon + dlon,
            "rain": float(v),
        })
        existing.add(name)
        idx += 1
    if syn_rows:
        df = pd.concat([df, pd.DataFrame(syn_rows)], ignore_index=True)
    return df

def add_scalebar(ax, location=(0.1, 0.06), linewidth=2, fontsize=9, length_km=10):
    """Dibuja una barra de escala fija en km (por defecto 10 km)."""
    proj = ccrs.PlateCarree()
    x0, x1, y0, y1 = ax.get_extent(crs=proj)
    lat = 0.5 * (y0 + y1)
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat))
    L = float(length_km)
    deg_len = L / km_per_deg_lon
    xa, ya = location
    lon0 = x0 + xa * (x1 - x0)
    lat0 = y0 + ya * (y1 - y0)
    lon1 = lon0 + deg_len
    ax.plot([lon0, lon1], [lat0, lat0], transform=proj, color="k", lw=linewidth, solid_capstyle="butt", zorder=11)
    ax.plot([lon0 + (lon1 - lon0) / 2.0, lon0 + (lon1 - lon0) / 2.0],
            [lat0 - 0.002*(y1-y0), lat0 + 0.002*(y1-y0)], transform=proj, color="k", lw=linewidth, zorder=11)
    ax.text((lon0 + lon1) / 2.0, lat0 - 0.03*(y1-y0), f"{L} km",
            ha="center", va="top", fontsize=fontsize, transform=proj, zorder=11)
