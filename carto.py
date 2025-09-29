import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Extensión Ecuador
ax.set_extent([-82, -74, -6, 4], crs=ccrs.PlateCarree())

# Relieve base
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

# Etiquetas principales
ax.text(-78.5, -0.2, "Quito", fontsize=10)
ax.text(-79.5, -2, "Puna", fontsize=10)
ax.text(-75.5, 1.5, "ASF", color='red', fontsize=12, fontweight='bold')

# Flechas tectónicas
ax.quiver(-79, -2, 0.2, 0.1, scale=2, color='k')  # dirección arbitraria
ax.text(-79.2, -2.2, "4-5 mm/yr", fontsize=8)

# Líneas de fallas (ejemplo)
lons = [-80, -78, -76]
lats = [-5, 0, 2]
ax.plot(lons, lats, 'r--', lw=1.2, label="Falla tectónica")

# Escala
ax.plot([-80, -79], [-5.5, -5.5], 'k-', lw=3)
ax.text(-79.8, -5.6, "100 km", fontsize=8)

# Grid y título
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4)
plt.title("Mapa tectónico del Ecuador y zonas de deformación")

plt.savefig("mapa_tectonico_ecuador.svg", dpi=300, bbox_inches='tight')
plt.show()
