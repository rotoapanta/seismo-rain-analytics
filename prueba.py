import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Extensión geográfica
ax.set_extent([-82, -77, -2, 2])  # Ecuador y costa

# Agregar topografía (batimetría simulada)
import cartopy.io.shapereader as shpreader
ax.add_feature(cfeature.LAND, zorder=1, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Grid
gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
gl.top_labels = gl.right_labels = False

# Puntos (ejemplo: estación y epicentro)
ax.plot(-78.49, -0.21, 'r^', markersize=8, label="Estación REVS2")
ax.plot(-79.5, 0.2, 'k*', markersize=10, label="Epicentro 2024")

# Anotación
ax.text(-78.6, -0.15, "REVS2", fontsize=9)
ax.text(-79.6, 0.25, "Evento", fontsize=9)

# Leyenda
plt.legend(loc='upper left')

plt.title("Mapa tectónico Ecuador")
plt.savefig("mapa_cartopy.svg", dpi=300, bbox_inches='tight')
plt.show()
