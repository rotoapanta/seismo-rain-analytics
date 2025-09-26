# NOTAS — Seismo-Rain Analytics

Este documento recopila notas operativas del proyecto: comandos útiles, parámetros del script de visualización, estructura de datos, y troubleshooting.

Última actualización: generar y mantener este archivo bajo control de versiones (git).

## Resumen
- Visualización de datos de estaciones sísmicas (SIS) y pluviómetro (RGA) desde archivos JSON en `DTA/`.
- Script principal: `main.py` (pandas + seaborn + matplotlib).
- Soporte de panel de mapa con coordenadas de estaciones (opcional, con o sin fondo cartográfico).

## Estructura de datos (JSON)
Ejemplo SIS (EC.REVS2.SIS_*.json):
```json
{
  "TIPO": "SIS",
  "NOMBRE": "REVS2",
  "IDENTIFICADOR": 1,
  "LECTURAS": [
    {
      "FECHA": "YYYY-MM-DD",
      "TIEMPO": "HH:MM:SS",
      "LATITUD": -0.212183,
      "LONGITUD": -78.491557,
      "ALTURA": 2814.1,
      "ALERTA": false,
      "PASA_BANDA": "0016",
      "PASA_BAJO": "0009",
      "PASA_ALTO": "0056",
      "BATERIA": 12.45
    }
  ]
}
```
Ejemplo RGA (EC.REVS2.RGA_*.json):
```json
{
  "TIPO": "RGA",
  "NOMBRE": "REVS2",
  "IDENTIFICADOR": 1,
  "LECTURAS": [
    {
      "FECHA": "YYYY-MM-DD",
      "TIEMPO": "HH:MM:SS",
      "LATITUD": -0.212183,
      "LONGITUD": -78.491557,
      "ALTURA": 2814.1,
      "NIVEL": 0.0,
      "BATERIA": 12.45
    }
  ]
}
```

## Comandos frecuentes
- Mostrar SIS con variables por defecto y mapa simple:
```bash
python main.py --tipo SIS --vars PASA_BANDA PASA_BAJO PASA_ALTO BATERIA --map --show
```
- Pluviómetro (RGA) con suavizado y mapa:
```bash
python main.py --tipo RGA --vars NIVEL BATERIA --smooth 7 --map --show
```
- Guardar sin mostrar (no abre ventana):
```bash
python main.py --tipo RGA --vars NIVEL --map --save reports/rga_con_mapa.png
```
- Fondo cartográfico (requiere dependencias, ver más abajo):
```bash
python main.py --tipo SIS --vars PASA_BANDA PASA_BAJO PASA_ALTO BATERIA \
  --map --basemap --basemap-provider "CartoDB.Positron" --show
```
- Varios puntos en el mapa (por archivo JSON) con separación para evitar superposición:
```bash
python main.py --map --hue SOURCE_FILE --map-jitter-m 25 \
  --tipo SIS --vars PASA_BANDA PASA_BAJO --show
```

## Parámetros de `main.py`
- `--root`: carpeta de datos (por defecto `DTA`).
- `--tipo`: `SIS` o `RGA` (filtrado por tipo de estación).
- `--nombre`: filtrar por nombre de estación exacto.
- `--vars`: variables a graficar.
  - SIS: `PASA_BANDA`, `PASA_BAJO`, `PASA_ALTO`, `BATERIA`.
  - RGA: `NIVEL`, `BATERIA`.
- `--hue`: columna para colorear series (por defecto `NOMBRE`).
- `--style`: columna para estilos de línea (opcional).
- `--smooth`: ventana de suavizado (media móvil centrada). `0/1` desactiva.
- `--markers`: activa marcadores en las líneas (por defecto desactivado).
- `--map`: añade panel de mapa a la derecha.
- `--basemap`: usa fondo de mapa (teselas) si hay dependencias.
- `--basemap-provider`: proveedor de teselas (p.ej., `OpenStreetMap.Mapnik`, `CartoDB.Positron`).
- `--map-jitter-m`: separa puntos coincidentes en el mapa (radio en metros).
- `--save`: ruta para guardar la imagen (crea directorios).
- `--dpi`: resolución al guardar (por defecto 130).
- `--show`: muestra la figura en pantalla.

Notas de visualización:
- Sin `--save`, `--show` abre ventana interactiva (puede bloquear hasta cerrar). Usa `Ctrl+C` para interrumpir.
- Si usas `--save` sin `--show`, el script usa el backend `Agg` y no abre ventana.

## Instalación y entorno
- Requisitos base: ver `requirements.txt` (pandas, seaborn, matplotlib, numpy).
- Crear entorno con `environment.yml` (si lo usas):
```bash
conda env create -f environment.yml
conda activate seismo-rain-analytics-env
```
- Instalar fondo cartográfico (opcional):
```bash
conda install -n seismo-rain-analytics-env -c conda-forge contextily pyproj
```
- Verificación rápida:
```bash
python -c "import contextily, pyproj; print('OK', contextily.__version__, pyproj.__version__)"
```

## Flujo de trabajo sugerido
1. Exportar/copiar los JSON a `DTA/` respetando la jerarquía por fecha.
2. Ejecutar `main.py` con filtros (`--tipo`, `--nombre`) y variables deseadas.
3. Ajustar `--smooth` para amortiguar ruido (p.ej., 5–11). Activar `--markers` si se requiere.
4. Activar `--map` para ver ubicación. Con varias estaciones usa `--hue NOMBRE` (por defecto ya lo hace) o `--hue SOURCE_FILE`.
5. Si hay múltiples estaciones en el mismo punto, usar `--map-jitter-m 10` (o más) para separarlas visualmente.
6. Guardar la figura con `--save reports/mi_figura.png`.

## Convenciones de archivos en `DTA/`
Ejemplo de nombre: `EC.REVS2.SIS_rpi-5_4513_20250925_1100.json`
- `EC`: país/código.
- `REVS2`: nombre de estación.
- `SIS`/`RGA`: tipo de estación.
- Sufijos: metadatos del dispositivo/serie.
- Fecha/hora final en `YYYYMMDD_HHMM`.

## Troubleshooting
- Warning de layout (tight_layout):
  - El script usa `constrained_layout=True` para evitar conflictos. No se usa `plt.tight_layout()`.
- Basemap no disponible (`ModuleNotFoundError`):
  - Instalar `contextily` y `pyproj` (ver sección de instalación). Si no hay internet o fallan, el mapa usa modo simple (sin fondo).
- Mapa muestra un solo punto:
  - Si solo hay una estación o `--hue` agrupa todo en un grupo, se dibuja un punto por grupo (estación). Para ver múltiples puntos:
    - Ejecutar sin filtro por `--nombre` si hay más estaciones.
    - Usar `--hue SOURCE_FILE` para un punto por archivo.
    - Añadir `--map-jitter-m 25` para separar puntos coincidentes.

## Backlog / Ideas
- [ ] Exportar figuras a PDF multipágina (una por variable).
- [ ] Soporte de tiles offline (mbtiles/local cache) para mapas sin internet.
- [ ] Métricas agregadas (resampling) por hora/día con barras/áreas.
- [ ] Panel de configuración (YAML) para listas de estaciones y estilos por defecto.
- [ ] Integrar validaciones de esquema JSON y reporte de errores.

## Registro de cambios breve
- 2025-09-25: `main.py` con suavizado, colores por variable, panel de mapa, metadatos en título.
- 2025-09-25: Añadidas opciones `--basemap`, `--basemap-provider`, `--map-jitter-m`.

```bash
rsync -avz pi@192.168.190.29:/home/pi/Documents/Projects/New/rain-gauge-project/DTA/ /home/rotoapanta/Documentos/Projects/seismo-rain-analytics/DTA/

rotoapanta@dsk-lnx:~/Documentos/Projects/seismo-rain-analytics$ python main.py --tipo SIS --vars PASA_BANDA PASA_BAJO PASA_ALTO BATERIA   --map --basemap --basemap-provider "CartoDB.Positron" --show

python main.py --tipo RGA --vars NIVEL BATERIA --map --basemap --map-zoom 18 --show

python main.py --tipo SIS --map --basemap --map-zoom 19 --show  

```