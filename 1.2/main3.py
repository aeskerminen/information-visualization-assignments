import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

# ── 1. Load & reshape ───────────────────────────────────────────────────────
df = pd.read_csv("sdg8_1.2.csv")
df.columns = [col.split(" ")[0] if "[" in col else col for col in df.columns]

df_long = df.melt(id_vars=["Country Name", "Series Name"],
                  var_name="Year", value_name="Value")
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["Year"]  = pd.to_numeric(df_long["Year"],  errors="coerce")
df_long = df_long.dropna()

df_pivot = df_long.pivot(index="Year", columns="Series Name",
                         values="Value").reset_index()

# Rename columns explicitly by matching the original Series Name strings
# (pivot sorts columns alphabetically, so don't assume order)
col_map = {
    "Year": "Year",
    "GDP (current US$)": "GDP_Trillion_USD",
    "Inflation, consumer prices (annual %)": "Inflation",
    "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)": "GHG_per_Capita",
    "Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)": "Youth_Unemployment",
}
df_pivot = df_pivot.rename(columns=col_map)

# Scale GDP to trillions for readable axis labels
df_pivot["GDP_Trillion_USD"] = df_pivot["GDP_Trillion_USD"] / 1e12

# Drop any year where data is incomplete (e.g. 2025 has no GDP/GHG/Inflation yet)
df_pivot = df_pivot.dropna(subset=["GHG_per_Capita", "GDP_Trillion_USD",
                                    "Inflation", "Youth_Unemployment"])

# ── 2. Continuous time colormap (viridis: 2000=dark-purple → 2025=yellow) ──
norm      = mcolors.Normalize(vmin=df_pivot["Year"].min(),
                               vmax=df_pivot["Year"].max())
cmap      = cm.viridis
point_colors = [cmap(norm(y)) for y in df_pivot["Year"]]

VARS  = ["GHG_per_Capita", "GDP_Trillion_USD", "Inflation", "Youth_Unemployment"]
LABELS = {
    "GHG_per_Capita":    "GHG Emissions\n(t CO₂e/capita)",
    "GDP_Trillion_USD":  "GDP\n(Trillion USD)",
    "Inflation":         "Inflation\n(annual %)",
    "Youth_Unemployment":"Youth Unemployment\n(%)",
}
n = len(VARS)

sns.set_theme(style="ticks", font_scale=0.95)
fig, axes = plt.subplots(n, n, figsize=(13, 12))
fig.suptitle(
    "Correlations and distributions of Global SDG 8 Indicators (2000–2025)\n",
    fontsize=13, y=1.01
)

for row, var_y in enumerate(VARS):
    for col, var_x in enumerate(VARS):
        ax = axes[row][col]

        if row == col:
            # ── Diagonal: histogram coloured by time decade ─────────────────
            ax.hist(df_pivot[var_x], bins=10,
                    color=cmap(0.5), edgecolor="white", linewidth=0.6)
            ax.set_facecolor("#f7f7f7")
            # Always show the variable name on the diagonal X-axis
            ax.set_xlabel(LABELS[var_x], fontsize=8)

        elif row > col:
            # ── Lower triangle: scatter + OLS regression line ───────────────
            x = df_pivot[var_x].values
            y = df_pivot[var_y].values

            ax.scatter(x, y, c=point_colors, s=40, zorder=3,
                       edgecolors="white", linewidths=0.4)

            # OLS fit
            slope, intercept, r, p, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="crimson", linewidth=1.4, zorder=4)
        else:
            # ── Upper triangle: large Pearson r as text (correlation matrix) ─
            x = df_pivot[var_x].values
            y = df_pivot[var_y].values
            r, p = stats.pearsonr(x, y)

            # Colour-code the cell by correlation strength
            cell_color = plt.cm.RdYlGn(0.5 + r / 2)   # −1→blue, 0→white, +1→red
            ax.set_facecolor(cell_color)
            ax.text(0.5, 0.5, f"r = {r:.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="black" if abs(r) < 0.7 else "white",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        # ── Axis labels on outer edges only ─────────────────────────────────
        if row == n - 1:
            ax.set_xlabel(LABELS[var_x], fontsize=8)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if col == 0:
            ax.set_ylabel(LABELS[var_y], fontsize=8)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.tick_params(axis="both", labelsize=7)

# ── 3. Shared colorbar for time ─────────────────────────────────────────────
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.02, pad=0.02, shrink=0.6)
cbar.set_label("Year", fontsize=10)
cbar.set_ticks([2000, 2005, 2010, 2015, 2020, 2025])

fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.15)
plt.savefig("sdg8_pairplot.png", dpi=150, bbox_inches="tight")
plt.show()
