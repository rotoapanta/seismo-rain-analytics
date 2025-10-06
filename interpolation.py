#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation and extent functions for seismo-rain-analytics project.
"""

import numpy as np
import pandas as pd

def idw_grid(lons, lats, vals, extent, nx=320, ny=280, power=2.0):
    xmin, xmax, ymin, ymax = extent
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    eps = 1e-9
    num = np.zeros_like(GX)
    den = np.zeros_like(GX)
    for lo, la, v in zip(lons, lats, vals):
        d2 = (GX - lo) ** 2 + (GY - la) ** 2
        w = 1.0 / (d2 + eps) ** (power / 2)
        num += w * v
        den += w
    return GX, GY, num / den

def auto_extent(df: pd.DataFrame, pad_deg=0.3):
    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    if lon_min == lon_max:
        lon_min -= 0.2; lon_max += 0.2
    if lat_min == lat_max:
        lat_min -= 0.2; lat_max += 0.2
    return [lon_min - pad_deg, lon_max + pad_deg, lat_min - pad_deg, lat_max + pad_deg]
