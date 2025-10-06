# Configuración única para main.py
# Usa este archivo como fuente de verdad y elimina los demás config.* si lo deseas.
# El script main.py prioriza este archivo si existe.

CONFIG = {
    # 1) Entradas de datos
    "dir": "DTA/2025/09/29/RGA",
    "date": "2025-09-29",
    "start": None,         # "HH:MM" o None
    "end": None,           # "HH:MM" o None
    "factor_tip": 1.0,     # factor mm/tip
    "recursive": False,    # buscar recursivamente

    # 2) Estaciones sintéticas
    "add_synthetic": False,
    "synth_mm": None,          # "5,12,25,40"
    "synth_offsets": None,     # "0.25,0.20; -0.35,0.15; 0.10,-0.30; -0.20,-0.25"
    "synth_base": "REVS2",

    # 3) Extent (lon_min, lon_max, lat_min, lat_max). None = automático
    "extent": None,

    # 4) Salida
    "out_prefix": None,    # prefijo de salida, None = auto
    "output_dir": "outputs/isoyetas",  # carpeta donde se guardan PNG/SVG

    # 5) Renderizado y fondo
    "scalebar_km": 10.0,
    # Desactiva OSM por defecto para evitar bloqueos por descarga de tiles
    "osm": False,
    "osm_zoom": None,
    "show_admin1_labels": True,
    "show_cities": True,
    "min_pop": 50000,

    # 6) Visualización general
    "no_show": False,
    # Tamaño de hoja y orientación
    "paper": "A3",               # "A3" | "A4" | None
    "orientation": "landscape",  # "landscape" | "portrait"

    # 7) Mapa principal: posición y tamaño
    # Opción A: posición manual en fracciones (0..1)
    "map_pos": [0.12, 0.15, 0.65, 0.70],
    # Opción B: márgenes y tamaño en cm (solo si map_pos es None)
    "left_margin_cm": 10.0,
    "bottom_margin_cm": None,
    "map_width_cm": 20.0,
    "map_height_cm": 20.0,

    # 8) Paneles inferiores y rosa
    "bottom_panels_height_cm": 5.0,
    "compass_image": "images/wind-rose.png",
    "compass_size": 0.38,

    # 9) Tamaño explícito de figura (opcional, en cm). Si se definen, tienen prioridad sobre paper/orientation
    "fig_width_cm": None,
    "fig_height_cm": None,
}
