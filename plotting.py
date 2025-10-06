#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting functions for seismo-rain-analytics project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io import img_tiles as cimgt
from cartopy.io import shapereader as shpreader
import matplotlib.ticker as mticker
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image
from cairosvg import svg2png

from utils import CM_PER_IN
from synthetic import add_scalebar
from interpolation import idw_grid

def plot_isoyetas(
    df,
    extent,
    fecha,
    rango_txt: str | None,
    out_prefix: str,
    scalebar_km: float = 10.0,
    use_osm: bool = False,
    osm_zoom: int | None = None,
    show_admin1_labels: bool = False,
    show_cities: bool = False,
    min_pop: int = 100000,
    show_plot: bool = True,
    fig_width: float | None = None,
    fig_height: float | None = None,
    left_margin_cm: float = 10.0,
    bottom_margin_cm: float | None = None,
    map_width_cm: float = 20.0,
    map_height_cm: float = 20.0,
    bottom_panels_height_cm: float = 5.0,
    compass_image: str | None = None,
    compass_size: float = 0.38,
    map_pos: list[float] | tuple[float, float, float, float] | None = None,
):
    proj = ccrs.PlateCarree()
    w = fig_width if fig_width else 10.5
    h = fig_height if fig_height else 8
    fig = plt.figure(figsize=(w, h))
    # Asegurar que la posición del eje no sea alterada por el sistema de layout
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    ax = plt.axes(projection=proj)
    ax.set_extent(extent, crs=proj)
    effective_osm = bool(use_osm) and bool(show_plot)
    if effective_osm:
        # Fondo con calles (OSM)
        try:
            tiler = cimgt.OSM()
            xmin, xmax, ymin, ymax = extent
            lat_c = 0.5 * (ymin + ymax)
            km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_c))
            width_km = (xmax - xmin) * km_per_deg_lon
            if osm_zoom is None:
                if width_km > 200:
                    z = 8
                elif width_km > 100:
                    z = 9
                elif width_km > 50:
                    z = 10
                elif width_km > 25:
                    z = 11
                elif width_km > 12:
                    z = 12
                elif width_km > 6:
                    z = 13
                else:
                    z = 14
            else:
                z = int(osm_zoom)
            ax.add_image(tiler, z)
        except Exception as e:
            print(f"[WARN] OSM deshabilitado por error: {e}. Usando fondo simple.")
            ax.add_feature(cfeature.OCEAN, facecolor="#dfefff")
            ax.add_feature(cfeature.LAND, facecolor="#f9f9f9")
    else:
        ax.add_feature(cfeature.OCEAN, facecolor="#dfefff")
        ax.add_feature(cfeature.LAND, facecolor="#f9f9f9")
    ax.add_feature(cfeature.COASTLINE, lw=0.7)
    ax.add_feature(cfeature.BORDERS, lw=0.6, linestyle=":")
    states = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', facecolor='none')
    ax.add_feature(states, edgecolor='0.35', linewidth=0.7, zorder=2)

    # Etiquetas de provincias (admin-1) opcionales, filtradas al extent y Ecuador
    if show_admin1_labels:
        try:
            shp_path = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
            reader = shpreader.Reader(shp_path)
            xmin, xmax, ymin, ymax = extent
            for rec in reader.records():
                attrs = rec.attributes
                admin = (attrs.get('admin') or attrs.get('ADMIN') or '').lower()
                iso3 = attrs.get('adm0_a3') or attrs.get('ADM0_A3') or ''
                if admin != 'ecuador' and iso3 != 'ECU':
                    continue
                geom = rec.geometry
                if geom is None:
                    continue
                pt = geom.representative_point()
                lon, lat = float(pt.x), float(pt.y)
                if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
                    continue
                name = attrs.get('name') or attrs.get('name_en') or attrs.get('name_local') or ''
                if not name:
                    continue
                ax.text(lon, lat, name, fontsize=7, color='0.15', weight='bold',
                        ha='center', va='center', transform=proj, zorder=12,
                        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.6))
        except Exception as _:
            pass

    title = f"Precipitación acumulada {fecha}"
    if rango_txt:
        title += f" ({rango_txt})"

    n = len(df)
    if n >= 3:
        lons = df["lon"].to_numpy(); lats = df["lat"].to_numpy(); vals = df["rain"].to_numpy()
        GX, GY, grid = idw_grid(lons, lats, vals, extent)
        # Escala fija Ecuador (acumulado diario en mm)
        bounds = np.array([0.0, 0.1, 5, 10, 20, 50, 100, 150, 200], dtype=float)
        # Asegurar número de colores >= número de bins (len(bounds)-1)
        n_bins = max(2, len(bounds) - 1)
        base_palette = ["#d4ebf2", "#a8d5e2", "#7dbdd2", "#55a4c2", "#2c8bb2", "#1f78a5", "#135c84", "#0b3f5d"]
        if len(base_palette) >= n_bins:
            cmap = ListedColormap(base_palette[:n_bins])
        else:
            # Genera una paleta discreta con el tamaño requerido
            cmap = plt.get_cmap("YlGnBu", n_bins)
        norm = BoundaryNorm(bounds, n_bins, clip=False)
        cf = ax.contourf(GX, GY, grid, levels=bounds, cmap=cmap, norm=norm,
                         transform=proj, alpha=(0.8 if use_osm else 0.95), extend="max")
        cs = ax.contour(GX, GY, grid, levels=bounds, colors="k", linewidths=0.45,
                        transform=proj, alpha=0.65)
        ax.clabel(cs, fmt=lambda v: f"{v:.1f}" if v < 1 else f"{v:.0f}", fontsize=7)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Precipitación (mm)")
        # Mantener 'title' solo con el texto principal; 'Isoyetas' será el título del eje
        title = title

        # Ciudades de referencia opcionales (umbral por población)
        if show_cities:
            try:
                shp_cities = shpreader.natural_earth(resolution='10m', category='cultural', name='populated_places')
                reader = shpreader.Reader(shp_cities)
                xmin, xmax, ymin, ymax = extent
                for rec in reader.records():
                    attrs = rec.attributes
                    lon = float(attrs.get('LONGITUDE', rec.geometry.x))
                    lat = float(attrs.get('LATITUDE', rec.geometry.y))
                    if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
                        continue
                    pop = int(attrs.get('POP_MAX') or attrs.get('POP_MIN') or 0)
                    if pop < int(min_pop):
                        continue
                    name = attrs.get('NAME') or attrs.get('NAMEASCII') or None
                    ax.scatter([lon], [lat], s=16, c=('white' if use_osm else 'k'), edgecolors='k',
                               linewidths=0.4, transform=proj, zorder=9)
                    if name:
                        ax.text(lon + 0.01*(xmax-xmin), lat + 0.005*(ymax-ymin), name,
                                fontsize=7, color='k', transform=proj, zorder=10,
                                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))
            except Exception as _:
                pass
    else:
        ax.text(0.5, 0.98,
                f"⚠️ {n} estación(es): no se pueden trazar isoyetas confiables.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="crimson")

    # Estaciones
    ax.scatter(df["lon"], df["lat"], marker="^", s=54, c="#ff0000",
               edgecolors="k", transform=proj, zorder=4, label="Estación")
    for _, r in df.iterrows():
        ax.text(r["lon"] + 0.03, r["lat"] + 0.03, f'{r["name"]} ({r["rain"]:.1f} mm)',
                fontsize=8, weight="bold", transform=proj)

    # Grilla + marco doble
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="0.25", alpha=0.6, linewidth=0.8)
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 10, "color": "0.2"}; gl.ylabel_style = {"size": 10, "color": "0.2"}
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    xmin, xmax, ymin, ymax = extent
    dx = xmax - xmin; dy = ymax - ymin
    step = 0.5 if max(dx, dy) < 5 else 1.0
    gl.xlocator = mticker.FixedLocator(np.arange(np.floor(xmin), np.ceil(xmax) + 1e-6, step))
    gl.ylocator = mticker.FixedLocator(np.arange(np.floor(ymin), np.ceil(ymax) + 1e-6, step))
    for i in [0.01, 0.018]:
        rect = mpatches.Rectangle((i, i), 1 - 2 * i, 1 - 2 * i,
                                  transform=ax.transAxes, fill=False, lw=1.2, ec="0.25", zorder=10, clip_on=False)
        ax.add_patch(rect)

    # Reubicar el mapa: si se provee map_pos (fracciones 0-1) se usa directamente;
    # en caso contrario, se calculan a partir de márgenes/tamaño en cm
    try:
        if map_pos is not None:
            left_frac, bottom_frac, width_frac, height_frac = map_pos
            ax.set_position([float(left_frac), float(bottom_frac), float(width_frac), float(height_frac)])
        else:
            pos = ax.get_position()
            fig_w_in = float(fig.get_size_inches()[0])
            fig_h_in = float(fig.get_size_inches()[1])
            fig_w_cm = fig_w_in * CM_PER_IN
            fig_h_cm = fig_h_in * CM_PER_IN
            # Fracciones en la figura
            left_frac = max(0.0, min(1.0, float(left_margin_cm) / fig_w_cm))
            bottom_frac = float(bottom_margin_cm) / fig_h_cm if bottom_margin_cm is not None else pos.y0
            bottom_frac = max(0.0, min(1.0, bottom_frac))
            width_frac = max(0.0, min(1.0, float(map_width_cm) / fig_w_cm))
            height_frac = max(0.0, min(1.0, float(map_height_cm) / fig_h_cm))
            # Asegurar que cabe en la figura
            width_frac = min(width_frac, 0.98 - left_frac)
            height_frac = min(height_frac, 0.98 - bottom_frac)
            ax.set_position([left_frac, bottom_frac, width_frac, height_frac])
    except Exception:
        pass

    # Tres cuadros iguales en la parte inferior (fuera del mapa) con doble margen
    ax_pos = ax.get_position()  # Bbox en coords de figura
    gap = 0.02  # separación
    # Altura disponible bajo el mapa
    available_h = max(0.0, ax_pos.y0 - 2 * gap)
    # Altura deseada de los cuadros inferiores en cm -> fracción de figura
    fig_h_in = float(fig.get_size_inches()[1])
    fig_h_cm = 2.54 * fig_h_in
    desired_h_frac = float(bottom_panels_height_cm) / fig_h_cm if fig_h_cm > 0 else 0.0
    # Usar la menor entre la deseada y la disponible para no solaparse con el mapa
    h = min(desired_h_frac, available_h)
    y0 = max(0.02, ax_pos.y0 - h - gap)
    # Ancho disponible alineado con el mapa
    total_w = ax_pos.x1 - ax_pos.x0
    w = (total_w - 2 * gap) / 3.0
    for k in range(3):
        x = ax_pos.x0 + k * (w + gap)
        for d in [0.0, 0.006]:
            frect = mpatches.Rectangle((x + d, y0 + d), w - 2 * d, h - 2 * d,
                                       transform=fig.transFigure, fill=False, lw=1.2, ec="0.25", zorder=1000, clip_on=False)
            fig.add_artist(frect)

    # SIMBOLOGIA en el primer recuadro inferior
    try:
        first_x = ax_pos.x0
        legend_pad = 0.006
        legend_ax = fig.add_axes([first_x + legend_pad, y0 + legend_pad, w - 2 * legend_pad, h - 2 * legend_pad], zorder=1100)
        legend_ax.set_axis_off()
        legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
        # Título y línea inferior
        legend_ax.text(0.5, 0.92, "SIMBOLOGÍA", ha="center", va="top", fontsize=10, weight="bold", color="0.1")
        legend_ax.plot([0.05, 0.95], [0.88, 0.88], color="0.25", lw=1.2)
        # Estación de monitoreo
        legend_ax.scatter([0.08], [0.70], marker="^", s=80, c="#ff0000", edgecolors="black", linewidths=0.6, zorder=2)
        legend_ax.text(0.16, 0.70, "Estación de monitoreo", va="center", fontsize=9, color="0.1")
        # Isoyeta (contorno)
        legend_ax.plot([0.06, 0.16], [0.54, 0.54], color="black", lw=1.2)
        legend_ax.text(0.16, 0.54, "Isoyeta (contorno)", va="center", fontsize=9, color="0.1")
        # Límites administrativos
        legend_ax.plot([0.06, 0.16], [0.40, 0.40], color="0.35", lw=1.0, linestyle=":")
        legend_ax.text(0.16, 0.40, "Límites administrativos", va="center", fontsize=9, color="0.1")
        # Nota de fondo
        legend_ax.text(0.06, 0.24, "Fondo: Calles (OSM)", va="center", fontsize=9, color="0.25")
    except Exception:
        pass

    # Minimap en el segundo recuadro inferior con título "MAPA DE UBICACION"
    try:
        second_x = ax_pos.x0 + (w + gap)
        mini_pad = 0.006
        mini_bounds = [second_x + mini_pad, y0 + mini_pad, w - 2 * mini_pad, h - 2 * mini_pad]
        ax_mini = plt.axes(mini_bounds, projection=proj, zorder=1100)
        # Configuración del minimapa
        # Extent fijo para Ecuador (ajustable si lo deseas)
        mini_extent = [-81.5, -75.0, -5.5, 2.5]
        ax_mini.set_extent(mini_extent, crs=proj)
        ax_mini.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax_mini.add_feature(cfeature.OCEAN, facecolor="#dfefff")
        ax_mini.add_feature(cfeature.COASTLINE, lw=0.4)
        ax_mini.add_feature(cfeature.BORDERS, lw=0.4, linestyle=":")
        # Rectángulo del extent principal
        xmin, xmax, ymin, ymax = extent
        dx = xmax - xmin; dy = ymax - ymin
        rect = mpatches.Rectangle((xmin, ymin), dx, dy, transform=proj, fill=False,
                                  ec="crimson", lw=1.2, zorder=1200)
        ax_mini.add_patch(rect)
        # Centro del extent
        cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax)
        ax_mini.scatter([cx], [cy], s=14, c="#ff0000", edgecolors="k", linewidths=0.4, transform=proj, zorder=1201)
        # Título del minimapa con línea
        ax_mini.text(0.5, 0.98, "MAPA DE UBICACION", transform=ax_mini.transAxes,
                     ha="center", va="top", fontsize=10, weight="bold", color="0.1")
        ax_mini.plot([0.05, 0.95], [0.94, 0.94], transform=ax_mini.transAxes, color="0.25", lw=1.0)
        # Ocultar ejes
        ax_mini.axis("off")
    except Exception:
        pass

    # Tercer recuadro inferior: Rosa de los vientos y escala del mapa
    try:
        third_x = ax_pos.x0 + 2 * (w + gap)
        panel_pad = 0.006
        panel_ax = fig.add_axes([third_x + panel_pad, y0 + panel_pad, w - 2 * panel_pad, h - 2 * panel_pad], zorder=1100)
        panel_ax.set_axis_off(); panel_ax.set_xlim(0, 1); panel_ax.set_ylim(0, 1)
        # Título y línea inferior del título
        panel_ax.text(0.5, 0.92, "ROSA DE LOS VIENTOS Y ESCALA", ha="center", va="top", fontsize=10, weight="bold", color="0.1")
        panel_ax.plot([0.05, 0.95], [0.88, 0.88], color="0.25", lw=1.2)
        # Rosa de los vientos a partir de imagen externa si se proporciona
        rendered_image = False
        if compass_image:
            try:
                path = compass_image
                if not os.path.isabs(path):
                    path = os.path.abspath(path)
                ext = os.path.splitext(path)[1].lower()
                img = None
                if ext == '.svg':
                    # Intentar rasterizar SVG a PNG en memoria usando cairosvg + PIL
                    try:
                        svg2png(bytestring=open(path, 'rb').read(), write_to=BytesIO())
                        buf = BytesIO()
                        svg2png(url=path, write_to=buf)
                        buf.seek(0)
                        im = Image.open(buf).convert('RGBA')
                        img = np.asarray(im)
                    except Exception as e:
                        print(f"[WARN] No se pudo rasterizar SVG '{path}': {e}. Usando rosa vectorial.")
                        img = None
                else:
                    try:
                        img = mpimg.imread(path)
                    except Exception as e:
                        print(f"[WARN] No se pudo leer imagen '{path}': {e}. Usando rosa vectorial.")
                        img = None
                if img is not None:
                    h_img, w_img = img.shape[0], img.shape[1]
                    cw = max(0.15, min(0.9, float(compass_size)))  # ancho relativo en coords del panel
                    ch = cw * (h_img / w_img)
                    cx, cy = 0.22, 0.58  # centro aproximado a la izquierda
                    x0, x1 = cx - cw / 2.0, cx + cw / 2.0
                    y0i, y1i = cy - ch / 2.0, cy + ch / 2.0
                    panel_ax.imshow(img, extent=(x0, x1, y0i, y1i), zorder=1101, aspect='auto')
                    rendered_image = True
                else:
                    rendered_image = False
            except Exception as e:
                print(f"[WARN] Error cargando rosa externa: {e}. Usando rosa vectorial.")
                rendered_image = False
        if not rendered_image:
            # Rosa de los vientos vectorial (fallback)
            cx, cy, r = 0.22, 0.55, 0.16
            circ = mpatches.Circle((cx, cy), r, fill=False, ec="0.2", lw=1.0)
            panel_ax.add_patch(circ)
            panel_ax.annotate("", xy=(cx, cy + r), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
            panel_ax.plot([cx - r, cx + r], [cy, cy], color="0.2", lw=0.8)
            panel_ax.plot([cx, cx], [cy - r, cy + r], color="0.2", lw=0.8)
            diag = r / np.sqrt(2)
            panel_ax.plot([cx - diag, cx + diag], [cy - diag, cy + diag], color="0.7", lw=0.6)
            panel_ax.plot([cx - diag, cx + diag], [cy + diag, cy - diag], color="0.7", lw=0.6)
            panel_ax.text(cx, cy + r + 0.04, "N", ha="center", va="bottom", fontsize=10, weight="bold")
            panel_ax.text(cx + r + 0.04, cy, "E", ha="left", va="center", fontsize=9)
            panel_ax.text(cx, cy - r - 0.04, "S", ha="center", va="top", fontsize=9)
            panel_ax.text(cx - r - 0.04, cy, "O", ha="right", va="center", fontsize=9)
        # Escala del mapa segmentada (derecha)
        sx0, sx1, sy = 0.50, 0.90, 0.24
        nseg = 4
        xs = np.linspace(sx0, sx1, nseg + 1)
        for i in range(nseg):
            col = 'k' if i % 2 == 0 else 'white'
            rect = mpatches.Rectangle((xs[i], sy - 0.015), xs[i+1] - xs[i], 0.03, transform=panel_ax.transAxes,
                                      facecolor=col, edgecolor='k', lw=0.8, zorder=1102)
            panel_ax.add_patch(rect)
        # Etiquetas 0..L, L=scalebar_km
        vals = [0, 0.25, 0.5, 0.75, 1.0]
        for frac, x in zip(vals, xs):
            val_km = frac * float(scalebar_km)
            lab = f"{int(round(val_km))}"
            panel_ax.text(x, sy - 0.06, lab, ha='center', va='top', fontsize=9)
        panel_ax.text(sx1 + 0.03, sy - 0.06, "Kilómetros", ha='left', va='top', fontsize=10)
    except Exception:
        pass

    # Recuadro en la derecha subdividido en 5 subcuadros (fuera del mapa)
    right_gap = 0.02
    right_w = min(0.08, max(0.05, 1.0 - ax_pos.x1 - 2 * right_gap))
    rx = min(1.0 - right_gap - right_w, ax_pos.x1 + right_gap)
    ry = ax_pos.y0
    rh = ax_pos.y1 - ax_pos.y0
    # Marco doble del recuadro principal
    for d in [0.0, 0.006]:
        frect = mpatches.Rectangle((rx + d, ry + d), right_w - 2 * d, rh - 2 * d,
                                   transform=fig.transFigure, fill=False, lw=1.2, ec="0.25", zorder=1000, clip_on=False)
        fig.add_artist(frect)
    # Subdividir verticalmente en 5 cajas iguales con doble borde
    sub_gap = 0.01 * rh
    sub_h = (rh - 4 * sub_gap) / 5.0
    for i_sub in range(5):
        sy = ry + i_sub * (sub_h + sub_gap)
        for d in [0.003, 0.006]:
            frect = mpatches.Rectangle((rx + d, sy + d), right_w - 2 * d, sub_h - 2 * d,
                                       transform=fig.transFigure, fill=False, lw=1.0, ec="0.25", zorder=1001, clip_on=False)
            fig.add_artist(frect)

    # Marco doble a toda la hoja (figura completa)
    for d in [0.004, 0.008]:
        frect = mpatches.Rectangle((d, d), 1 - 2 * d, 1 - 2 * d,
                                   transform=fig.transFigure, fill=False, lw=1.4, ec="0.25", zorder=2000, clip_on=False)
        fig.add_artist(frect)

    add_scalebar(ax, length_km=scalebar_km)
    # Título del eje y título principal de la figura
    ax.set_title("Isoyetas", pad=6, fontsize=12, weight="bold")
    fig.suptitle(title, y=0.99, fontsize=16, weight="bold")
    png = f"{out_prefix}.png"; svg = f"{out_prefix}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    print(f"[OK] Exportado: {png} | {svg}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
