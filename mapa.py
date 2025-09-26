#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de mapa con un waypoint. Si hay contextily+pyproj, dibuja fondo OSM.
Si no, cae a un scatter simple en Lat/Lon (sin dependencias extra).

Ejecuta:
  python test_map.py --show
  python test_map.py --save reporte.png
"""

import argparse
import matplotlib.pyplot as plt

# Tu waypoint
LAT = -0.212183
LON = -78.491557
ALT = 2814.1

def plot_simple(ax):
    ax.set_title("Waypoint (modo simple Lat/Lon)")
    ax.scatter([LON], [LAT], s=80, edgecolors="black")  # punto
    ax.annotate(f"ALT {ALT:.1f} m", (LON, LAT), xytext=(5, 5),
                textcoords="offset points", fontsize=9)
    # margen alrededor del punto
    pad_deg = 0.01
    ax.set_xlim(LON - pad_deg, LON + pad_deg)
    ax.set_ylim(LAT - pad_deg, LAT + pad_deg)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

def plot_with_basemap(ax, provider_str="CartoDB.Positron", zoom_to_point=True):
    from pyproj import Transformer
    import contextily as ctx

    # Proyección a Web Mercator (EPSG:3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(LON, LAT)

    ax.set_title(f"Waypoint con basemap ({provider_str})")
    ax.scatter([x], [y], s=80, edgecolors="black")  # punto en 3857
    ax.annotate(f"ALT {ALT:.1f} m", (x, y), xytext=(5, 5),
                textcoords="offset points", fontsize=9)

    # Extensión alrededor del punto (en metros)
    if zoom_to_point:
        pad_m = 500  # ajusta zoom (500 m a cada lado)
        ax.set_xlim(x - pad_m, x + pad_m)
        ax.set_ylim(y - pad_m, y + pad_m)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2, linestyle=":")

    # proveedor de teselas
    src = ctx.providers.CartoDB.Positron
    try:
        cur = ctx.providers
        for part in provider_str.split("."):
            cur = getattr(cur, part)
        src = cur
    except Exception:
        pass

    ctx.add_basemap(ax, source=src)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", type=str, default=None, help="Ruta para guardar la figura")
    p.add_argument("--show", action="store_true", help="Mostrar en pantalla")
    p.add_argument("--provider", type=str, default="CartoDB.Positron",
                   help="Proveedor basemap: OpenStreetMap.Mapnik, CartoDB.Positron, CartoDB.DarkMatter, etc.")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))
    used_basemap = False
    try:
        # intenta basemap
        plot_with_basemap(ax, provider_str=args.provider)
        used_basemap = True
    except Exception as e:
        ax.text(0.5, 0.98, f"Basemap no disponible: {type(e).__name__}",
                ha="center", va="top", transform=ax.transAxes, fontsize=9)
        plot_simple(ax)

    fig.suptitle(f"Waypoint @ ({LAT:.6f}, {LON:.6f}) — ALT {ALT:.1f} m", y=0.98)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"[OK] Figura guardada en {args.save}")

    if args.show or not args.save:
        plt.show()

    print("[INFO] Modo:", "basemap" if used_basemap else "simple")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
