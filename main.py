#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("DTA")
DEFAULT_VARS = ["PASA_BANDA", "PASA_BAJO", "PASA_ALTO", "BATERIA"]
NUMERIC_CANDIDATES = [
    "PASA_BANDA", "PASA_BAJO", "PASA_ALTO", "NIVEL",   # <-- añadido NIVEL
    "BATERIA", "LATITUD", "LONGITUD", "ALTURA"
]


def find_json_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def load_json_records(paths: Iterable[Path]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for fp in paths:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] No se pudo leer {fp}: {e}")
            continue

        tipo = data.get("TIPO")
        nombre = data.get("NOMBRE")
        identificador = data.get("IDENTIFICADOR")
        lecturas = data.get("LECTURAS", []) or []

        for row in lecturas:
            r = dict(row)
            r["TIPO"] = tipo
            r["NOMBRE"] = nombre
            r["IDENTIFICADOR"] = identificador
            r["SOURCE_FILE"] = str(fp)
            records.append(r)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Parse de fecha/hora
    if {"FECHA", "TIEMPO"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["FECHA"].astype(str).str.strip() + " " + df["TIEMPO"].astype(str).str.strip(),
            errors="coerce",
            format="%Y-%m-%d %H:%M:%S",
        )
    elif "FECHA" in df and "TIEMPO" not in df:
        df["timestamp"] = pd.to_datetime(df["FECHA"], errors="coerce")

    # Conversión a numérico donde aplique
    for c in NUMERIC_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Orden
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def apply_rolling_smoothing(df: pd.DataFrame, variables: List[str], window: int, group_keys: List[str] | None = None) -> pd.DataFrame:
    if window is None or window <= 1:
        return df

    out = df.copy()
    group_keys = [g for g in (group_keys or []) if g in out.columns]

    if "timestamp" in out.columns:
        sort_cols = [c for c in group_keys] + ["timestamp"] if group_keys else ["timestamp"]
        out = out.sort_values(sort_cols)

    minp = max(1, window // 2)
    present_vars = [v for v in variables if v in out.columns]
    if not present_vars:
        return out

    for var in present_vars:
        if group_keys:
            out[var] = (
                out.groupby(group_keys, sort=False)[var]
                .transform(lambda s: s.rolling(window=window, center=True, min_periods=minp).mean())
            )
        else:
            out[var] = out[var].rolling(window=window, center=True, min_periods=minp).mean()

    return out


def plot_map_panel(
    df: pd.DataFrame,
    ax: plt.Axes,
    hue_by: str | None = "NOMBRE",
    palette_map: Dict[str, Any] | None = None,
    basemap: bool = False,
    basemap_provider: str = "OpenStreetMap.Mapnik",
    jitter_m: float = 0.0,
    map_zoom: Optional[int] = None,          # <-- nuevo
):
    """Dibuja un panel de mapa con coordenadas de estaciones."""
    if "LATITUD" not in df.columns or "LONGITUD" not in df.columns:
        ax.text(0.5, 0.5, "Sin coordenadas", ha="center", va="center")
        ax.set_axis_off()
        return

    dff = df.dropna(subset=["LATITUD", "LONGITUD"]).copy()
    if dff.empty:
        ax.text(0.5, 0.5, "Sin coordenadas", ha="center", va="center")
        ax.set_axis_off()
        return

    # Construcción de puntos etiquetados
    pts: List[tuple[str, float, float]] = []
    if hue_by and hue_by in dff.columns:
        gg = dff.groupby(hue_by, as_index=False).agg({"LATITUD": "mean", "LONGITUD": "mean"})
        for _, r in gg.iterrows():
            pts.append((str(r[hue_by]), float(r["LATITUD"]), float(r["LONGITUD"])))
    else:
        uniq = dff[["LATITUD", "LONGITUD"]].drop_duplicates().reset_index(drop=True)
        for i, r in uniq.iterrows():
            pts.append((f"P{i+1}", float(r["LATITUD"]), float(r["LONGITUD"])))

    # Extensión geográfica
    lats = [p[1] for p in pts]
    lons = [p[2] for p in pts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Jitter de puntos
    if jitter_m and jitter_m > 0:
        from collections import defaultdict
        loc_groups: Dict[tuple, List[int]] = defaultdict(list)
        for idx, (_, lat, lon) in enumerate(pts):
            key = (round(lat, 6), round(lon, 6))
            loc_groups[key].append(idx)
        lat0 = float(np.mean(lats))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
        r_deg_lat = jitter_m / m_per_deg_lat
        r_deg_lon = jitter_m / m_per_deg_lon if m_per_deg_lon > 0 else 0.0
        for group in loc_groups.values():
            if len(group) <= 1:
                continue
            k = len(group)
            for j, idx in enumerate(group):
                angle = 2 * np.pi * j / k
                name, lat, lon = pts[idx]
                pts[idx] = (name, lat + r_deg_lat * np.sin(angle), lon + r_deg_lon * np.cos(angle))
        lats = [p[1] for p in pts]
        lons = [p[2] for p in pts]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

    # Preparación del eje
    ax.set_title("Mapa de estaciones")
    ax.grid(True, alpha=0.3, linestyle="--")

    default_color = "#1f77b4"

    if basemap:
        try:
            from pyproj import Transformer
            import contextily as ctx

            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xy = [transformer.transform(lon, lat) for (_, lat, lon) in pts]
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            dx = max(1.0, max_x - min_x)
            dy = max(1.0, max_y - min_y)
            pad = max(dx, dy) * 0.30 + 200.0  # un poco más de margen

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(min_x - pad, max_x + pad)
            ax.set_ylim(min_y - pad, max_y + pad)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            # Proveedor
            src = ctx.providers.OpenStreetMap.Mapnik
            try:
                cur = ctx.providers
                for part in basemap_provider.split('.'):
                    cur = getattr(cur, part)
                src = cur
            except Exception:
                pass

            # Zoom seguro (evita warning de 21 en proveedores 0–20)
            if map_zoom is not None:
                safe_zoom = max(0, min(int(map_zoom), 20))
                ctx.add_basemap(ax, source=src, zoom=safe_zoom)
            else:
                # inferido por contextily; ya dimos más padding para evitar >20
                ctx.add_basemap(ax, source=src)

            for (name, _lat, _lon), x, y in zip(pts, xs, ys):
                color = palette_map.get(name) if (palette_map and name in palette_map) else default_color
                ax.scatter(x, y, s=65, c=[color], edgecolors="black", linewidths=0.7, zorder=3)
                ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)
            return
        except Exception as e:
            ax.text(0.5, 0.98, f"Basemap no disponible: {type(e).__name__}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
            # caer a modo simple

    # Modo simple (Lat/Lon)
    dlat = max(1e-6, max_lat - min_lat)
    dlon = max(1e-6, max_lon - min_lon)
    pad_deg = max(dlat, dlon) * 0.15 + 0.01
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_lon - pad_deg, max_lon + pad_deg)
    ax.set_ylim(min_lat - pad_deg, max_lat + pad_deg)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    for name, lat, lon in pts:
        color = palette_map.get(name) if (palette_map and name in palette_map) else default_color
        ax.scatter(lon, lat, s=65, c=[color], edgecolors="black", linewidths=0.7, zorder=3)
        ax.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points", fontsize=9)


def plot_time_series(
    df: pd.DataFrame,
    variables: List[str],
    hue_by: str = "NOMBRE",
    style_by: str | None = None,
    height: float = 2.6,
    aspect: float = 3.0,
    markers: bool = False,
    add_map: bool = False,
    map_basemap: bool = False,
    map_provider: str = "OpenStreetMap.Mapnik",
    map_jitter_m: float = 0.0,
    map_zoom: Optional[int] = None,          # <-- nuevo
):
    present_vars = [v for v in variables if v in df.columns]
    if not present_vars:
        raise ValueError(f"Ninguna de las variables solicitadas está presente: {variables}")

    sns.set_theme(style="whitegrid", context="talk")

    n = len(present_vars)
    if add_map:
        fig = plt.figure(figsize=(aspect * 3.2 * 1.3, height * n), constrained_layout=True)
        gs = fig.add_gridspec(nrows=n, ncols=2, width_ratios=[3.0, 1.2])
        axes_ts = [fig.add_subplot(gs[i, 0]) for i in range(n)]
        ax_map = fig.add_subplot(gs[:, 1])
    else:
        fig, axes_ts = plt.subplots(n, 1, figsize=(aspect * 3.2, height * n), sharex=True, constrained_layout=True)
        if n == 1:
            axes_ts = [axes_ts]
        ax_map = None

    hue_param_global = hue_by if hue_by in df.columns else None
    levels_sorted_global = None
    if hue_param_global:
        levels_sorted_global = sorted({str(x) for x in df[hue_param_global].dropna().unique()})

    base_colors = sns.color_palette("tab10")
    for i, (ax, var) in enumerate(zip(axes_ts, present_vars)):
        hue_param = hue_by if hue_by in df.columns else None
        style_param = style_by if style_by and style_by in df.columns else None

        palette_kwargs = {}
        if hue_param:
            levels = list(pd.Series(df[hue_param]).dropna().unique())
            if len(levels) > 1:
                levels_sorted = sorted(levels, key=lambda x: str(x))
                palette_map = {lvl: base_colors[(i + j) % len(base_colors)] for j, lvl in enumerate(levels_sorted)}
                palette_kwargs = {"palette": palette_map}
            else:
                palette_kwargs = {"color": base_colors[i % len(base_colors)]}
                hue_param = None
        else:
            palette_kwargs = {"color": base_colors[i % len(base_colors)]}

        sns.lineplot(
            data=df,
            x="timestamp",
            y=var,
            hue=hue_param,
            style=style_param,
            marker="o" if markers else None,
            ax=ax,
            linewidth=1.6,
            errorbar=None,
            **palette_kwargs,
        )
        ax.set_title(var)
        ax.set_xlabel("")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        if "timestamp" in df.columns:
            fig.autofmt_xdate(rotation=20)

    if add_map and ax_map is not None:
        palette_map_global = None
        if hue_param_global and levels_sorted_global:
            palette_map_global = {lvl: base_colors[j % len(base_colors)] for j, lvl in enumerate(levels_sorted_global)}
        plot_map_panel(
            df,
            ax_map,
            hue_by=hue_param_global,
            palette_map=palette_map_global,
            basemap=map_basemap,
            basemap_provider=map_provider,
            jitter_m=map_jitter_m,
            map_zoom=map_zoom,          # <-- pasa el zoom
        )

    try:
        rango = ""
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            tmin = df["timestamp"].min()
            tmax = df["timestamp"].max()
            rango = f" — {tmin:%Y-%m-%d %H:%M:%S} a {tmax:%Y-%m-%d %H:%M:%S}"
        titulo = f"Seismo-Rain: {', '.join(present_vars)}{rango}"
        fig.suptitle(titulo, y=0.98, fontsize=14)

        info_parts = []
        if "NOMBRE" in df.columns and df["NOMBRE"].notna().any():
            nombres_unique = sorted({str(x) for x in df["NOMBRE"].dropna().unique()})
            label = "Estación" if len(nombres_unique) == 1 else "Estaciones"
            info_parts.append(f"{label}: {', '.join(nombres_unique)}")
        if "TIPO" in df.columns and df["TIPO"].notna().any():
            tipos_unique = sorted({str(x) for x in df["TIPO"].dropna().unique()})
            label = "Tipo" if len(tipos_unique) == 1 else "Tipos"
            info_parts.append(f"{label}: {', '.join(tipos_unique)}")
        if "IDENTIFICADOR" in df.columns and df["IDENTIFICADOR"].notna().any():
            ids_unique = sorted({str(x) for x in df["IDENTIFICADOR"].dropna().unique()})
            if len(ids_unique) <= 3:
                label = "ID" if len(ids_unique) == 1 else "IDs"
                info_parts.append(f"{label}: {', '.join(ids_unique)}")

        lat = lon = None
        if "LATITUD" in df.columns and df["LATITUD"].notna().any():
            lat_u = pd.unique(df["LATITUD"].dropna().round(6))
            if len(lat_u) == 1:
                lat = float(lat_u[0])
        if "LONGITUD" in df.columns and df["LONGITUD"].notna().any():
            lon_u = pd.unique(df["LONGITUD"].dropna().round(6))
            if len(lon_u) == 1:
                lon = float(lon_u[0])
        if lat is not None and lon is not None:
            info_parts.append(f"Lat,Lon: {lat:.6f}, {lon:.6f}")

        if "SOURCE_FILE" in df.columns:
            info_parts.append(f"Archivos: {df['SOURCE_FILE'].nunique()}")
        info_parts.append(f"Muestras: {len(df)}")

        subtitulo = " — ".join(info_parts)
        fig.text(0.5, 0.955, subtitulo, ha="center", va="top", fontsize=10, alpha=0.9)
    except Exception:
        pass

    for ax in axes_ts:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", frameon=True)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Graficar lecturas JSON con seaborn")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Carpeta raíz de datos (por defecto: DTA)")
    parser.add_argument("--tipo", type=str, choices=["SIS", "RGA"], default=None, help="Filtrar por TIPO")
    parser.add_argument("--nombre", type=str, default=None, help="Filtrar por NOMBRE exacto")
    parser.add_argument("--vars", nargs="+", default=DEFAULT_VARS, help=f"Variables a graficar (por defecto: {' '.join(DEFAULT_VARS)})")
    parser.add_argument("--hue", type=str, default="NOMBRE", help="Columna para hue (por defecto: NOMBRE)")
    parser.add_argument("--style", type=str, default=None, help="Columna para style (opcional)")
    parser.add_argument("--smooth", type=int, default=5, help="Ventana de suavizado (media móvil centrada). 0/1 desactiva")
    parser.add_argument("--markers", action="store_true", help="Mostrar marcadores en las líneas")
    parser.add_argument("--map", action="store_true", help="Añadir un panel de mapa con coordenadas a la derecha")
    parser.add_argument("--basemap", action="store_true", help="Usar fondo de mapa (requiere contextily y pyproj)")
    parser.add_argument("--basemap-provider", type=str, default="OpenStreetMap.Mapnik", help="Proveedor de teselas (p.ej., OpenStreetMap.Mapnik, CartoDB.Positron)")
    parser.add_argument("--map-jitter-m", type=float, default=0.0, help="Separar puntos coincidentes en el mapa (radio en metros)")
    parser.add_argument("--map-zoom", type=int, default=19, help="Zoom del basemap (0-20 para CartoDB/OSM)")  # <-- nuevo
    parser.add_argument("--save", type=str, default=None, help="Ruta para guardar la figura (si no se especifica, no se guarda)")
    parser.add_argument("--dpi", type=int, default=130, help="DPI al guardar la imagen")
    parser.add_argument("--show", action="store_true", help="Mostrar la figura en pantalla")

    args = parser.parse_args()

    root = Path(args.root)
    files = find_json_files(root)
    if not files:
        print(f"[ERROR] No se encontraron JSON en {root}")
        return 2

    df = load_json_records(files)
    if df.empty:
        print("[ERROR] No se pudo construir el DataFrame a partir de los JSON")
        return 2

    # Filtros
    if args.tipo:
        df = df[df.get("TIPO").eq(args.tipo)]
    if args.nombre:
        df = df[df.get("NOMBRE").eq(args.nombre)]

    if df.empty:
        print("[WARN] No hay datos luego de aplicar filtros")
        return 0

    if args.save and not args.show:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    df_plot = apply_rolling_smoothing(df, variables=args.vars, window=args.smooth, group_keys=[args.hue, args.style])

    fig = plot_time_series(
        df=df_plot,
        variables=args.vars,
        hue_by=args.hue,
        style_by=args.style,
        markers=args.markers,
        add_map=args.map,
        map_basemap=args.basemap,
        map_provider=args.basemap_provider,
        map_jitter_m=args.map_jitter_m,
        map_zoom=args.map_zoom,          # <-- pasa el zoom
    )

    if args.save:
        outpath = Path(args.save)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        print(f"[INFO] Figura guardada en: {outpath}")

    if args.show or not args.save:
        try:
            plt.show()
        except KeyboardInterrupt:
            print("[WARN] Visualización interrumpida por el usuario.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
