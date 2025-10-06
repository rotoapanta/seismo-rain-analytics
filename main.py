#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera un mapa de precipitación (isoyetas) a partir de archivos JSON RCAG.

- Lee todos los JSON en un directorio (por defecto DTA/2025/09/29/RGA)
- Acumula la lluvia (campo NIVEL) por estación en una fecha y rango horario opcional
- Usa las coordenadas LATITUD y LONGITUD que vienen en los JSON
- Interpola por IDW y exporta PNG y SVG
- Permite añadir 4 estaciones sintéticas alrededor de REVS2 para complementar
  una red escasa y poder trazar isoyetas

Uso:
  python main.py \
    --dir DTA/2025/09/29/RGA \
    --date 2025-09-29 \
    --start 16:00 --end 16:59 \
    --factor-tip 1.0 \
    --extent -79.9 -77.4 -2.0 1.2 \
    --recursive \
    --add-synthetic --synth-mm '5,12,25,40' \
    --synth-offsets '0.25,0.20; -0.35,0.15; 0.10,-0.30; -0.20,-0.25' \
    --synth-base REVS2

Si no se proporciona extent, se calcula automáticamente a partir del bounding box
de las estaciones (con margen).
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import pandas as pd
from datetime import date

from utils import A3_SIZE_CM, A4_SIZE_CM, cm_to_in, parse_date, parse_time
from data_loader import cargar_rcag
from interpolation import auto_extent
from synthetic import add_synthetic
from plotting import plot_isoyetas

def build_argparser():
    p = argparse.ArgumentParser(description="Mapa de isoyetas desde JSON RCAG")
    p.add_argument("--dir", default="DTA/2025/09/29/RGA", help="Directorio con JSON (RCAG)")
    p.add_argument("--date", default="2025-09-29", help="Fecha YYYY-MM-DD a acumular")
    p.add_argument("--start", default=None, help="Hora inicio HH:MM (opcional)")
    p.add_argument("--end", default=None, help="Hora fin HH:MM (opcional)")
    p.add_argument("--factor-tip", type=float, default=1.0, help="Factor mm/tip (p.ej. 0.2)")
    p.add_argument("--recursive", action="store_true", help="Buscar JSON recursivamente dentro del directorio")
    p.add_argument("--add-synthetic", action="store_true", help="Agregar 4 estaciones sintéticas alrededor de la base (REVS2 por defecto)")
    p.add_argument("--synth-mm", default=None, help="Lista de mm para sintéticas, p.ej. '5,12,25,40'")
    p.add_argument("--synth-offsets", default=None, help="Lista de offsets dlon,dlat separados por ';', p.ej. '0.25,0.20; -0.35,0.15; 0.10,-0.30; -0.20,-0.25'")
    p.add_argument("--synth-base", default="REVS2", help="Nombre de estación base para posicionar las sintéticas")
    p.add_argument("--extent", nargs=4, type=float, default=None,
                   help="Extent lon_min lon_max lat_min lat_max (opcional)")
    p.add_argument("--out-prefix", default=None,
                   help="Prefijo de salida (opcional). Por defecto se genera uno automático")
    p.add_argument("--scalebar-km", type=float, default=10.0,
                   help="Longitud de la barra de escala en km (por defecto 10)")
    p.add_argument("--osm", action="store_true", default=True,
                   help="Usar fondo OSM (calles) (por defecto ACTIVADO)")
    p.add_argument("--osm-zoom", type=int, default=None,
                   help="Nivel de zoom OSM (auto si no se especifica)")
    p.add_argument("--show-admin1-labels", action="store_true", default=True,
                   help="Mostrar etiquetas de provincias (admin-1) dentro del extent (por defecto ACTIVADO)")
    p.add_argument("--show-cities", action="store_true", default=True,
                   help="Mostrar ciudades de referencia (Natural Earth) (por defecto ACTIVADO)")
    p.add_argument("--min-pop", type=int, default=50000,
                   help="Umbral mínimo de población para etiquetar ciudades (por defecto 50000)")
    p.add_argument("--no-show", action="store_true",
                   help="No abrir ventana interactiva (útil en servidores/automatización)")
    p.add_argument("--paper", choices=["A3", "A4"], default=None,
                   help="Tamaño de hoja en pulgadas (A3=16.54x11.69 landscape, 11.69x16.54 portrait; A4=11.69x8.27 landscape, 8.27x11.69 portrait)")
    p.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape",
                   help="Orientación de la hoja cuando se usa --paper")
    p.add_argument("--left-margin-cm", type=float, default=10.0,
                   help="Margen izquierdo del mapa desde el borde de la hoja, en cm (por defecto 10)")
    p.add_argument("--bottom-margin-cm", type=float, default=None,
                   help="Margen inferior del mapa desde el borde de la hoja, en cm (opcional)")
    p.add_argument("--map-width-cm", type=float, default=20.0,
                   help="Ancho del mapa (eje principal) en cm (por defecto 20)")
    p.add_argument("--map-height-cm", type=float, default=20.0,
                   help="Alto del mapa (eje principal) en cm (por defecto 20)")
    p.add_argument("--compass-image", default="images/wind-rose.png",
                   help="Ruta a imagen de rosa de los vientos para el tercer recuadro (PNG/SVG/JPG). Por defecto images/wind-rose.png")
    p.add_argument("--compass-size", type=float, default=0.38,
                   help="Ancho relativo (0-1) de la imagen de rosa en el tercer recuadro (por defecto 0.38)")
    p.add_argument("--bottom-panels-height-cm", type=float, default=5.0,
                   help="Altura de cada uno de los tres cuadros inferiores, en cm (por defecto 5)")
    p.add_argument("--map-pos", nargs=4, type=float, default=None,
                   metavar=("LEFT","BOTTOM","WIDTH","HEIGHT"),
                   help="Posición del mapa en coordenadas de figura (fracciones 0-1): left bottom width height")

    return p

def main():
    # For CLI mode, parse arguments
    parser = build_argparser()
    args = parser.parse_args()

    dpath = Path(args.dir)
    dia = parse_date(args.date)
    t0 = parse_time(args.start) if args.start else None
    t1 = parse_time(args.end) if args.end else None
    factor_tip = float(args.factor_tip)

    df = cargar_rcag(dpath, dia, t0, t1, factor_tip, recursive=args.recursive)
    if df.empty:
        print(f"[ERROR] No se encontraron datos en {dpath} para {dia} (rango: {args.start}-{args.end})")
        return

    # Agregar estaciones sintéticas si se solicita o si hay pocas estaciones
    auto_added = False
    if args.add_synthetic or len(df) < 3:
        if len(df) < 3 and not args.add_synthetic:
            print(f"[INFO] {len(df)} estación(es). Agregando estaciones sintéticas para poder trazar isoyetas.")
            auto_added = True
        df = add_synthetic(df, base_name=args.synth_base,
                           offsets_str=args.synth_offsets,
                           values_str=args.synth_mm)

    # Extent
    if args.extent is not None:
        extent = list(map(float, args.extent))
    else:
        extent = auto_extent(df, pad_deg=0.5)

    # Prefijo de salida
    rango_txt = None
    if t0 and t1:
        rango_txt = f"{t0.strftime('%H%M')}-{t1.strftime('%H%M')}"
    out_prefix = args.out_prefix
    if not out_prefix:
        out_prefix = f"isoyetas_rcag_{dia}"
        if rango_txt:
            out_prefix += f"_{rango_txt}"
    # Directorio de salida para isoyetas
    output_dir = args.output_dir if hasattr(args, 'output_dir') else "outputs/isoyetas"
    out_dir_path = Path(output_dir)
    try:
        out_dir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARN] No se pudo crear directorio de salida '{out_dir_path}': {e}")
    else:
        out_prefix = str(out_dir_path / out_prefix)

    # Resumen CSV
    df.to_csv(f"acumulado_rcag_{dia}.csv", index=False)

    # Plot
    fig_w = None; fig_h = None
    if args.paper:
        if args.paper == "A3":
            w_cm, h_cm = A3_SIZE_CM
        elif args.paper == "A4":
            w_cm, h_cm = A4_SIZE_CM
        else:
            w_cm, h_cm = A3_SIZE_CM
        if args.orientation == "landscape":
            fig_w, fig_h = (cm_to_in(w_cm), cm_to_in(h_cm))
        else:
            fig_w, fig_h = (cm_to_in(h_cm), cm_to_in(w_cm))
    # Override con dimensiones explícitas de figura si se proporcionan
    fig_width_cm = args.fig_width_cm if hasattr(args, 'fig_width_cm') else None
    fig_height_cm = args.fig_height_cm if hasattr(args, 'fig_height_cm') else None
    if fig_width_cm is not None or fig_height_cm is not None:
        wcm = float(fig_width_cm) if fig_width_cm is not None else (w_cm if 'w_cm' in locals() else 27.0)
        hcm = float(fig_height_cm) if fig_height_cm is not None else (h_cm if 'h_cm' in locals() else 20.0)
        fig_w, fig_h = (cm_to_in(wcm), cm_to_in(hcm))
    plot_isoyetas(df, extent, dia, rango_txt, out_prefix,
                  scalebar_km=float(args.scalebar_km),
                  use_osm=bool(args.osm),
                  osm_zoom=args.osm_zoom,
                  show_admin1_labels=bool(args.show_admin1_labels),
                  show_cities=bool(args.show_cities),
                  min_pop=int(args.min_pop),
                  show_plot=(not args.no_show),
                  fig_width=fig_w,
                  fig_height=fig_h,
                  left_margin_cm=float(args.left_margin_cm),
                  bottom_margin_cm=(float(args.bottom_margin_cm) if args.bottom_margin_cm is not None else None),
                  map_width_cm=float(args.map_width_cm),
                  map_height_cm=float(args.map_height_cm),
                  bottom_panels_height_cm=float(args.bottom_panels_height_cm),
                  compass_image=args.compass_image,
                  compass_size=float(args.compass_size),
                  map_pos=args.map_pos)

if __name__ == "__main__":
    main()
