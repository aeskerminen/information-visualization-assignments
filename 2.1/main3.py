import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

df = pd.read_csv("youth_unemployment_2000_2025_majorregions.csv")
df.columns = [col.split(" ")[0] if "[" in col else col for col in df.columns]

df_long = df.melt(id_vars=["Country Name"],
                  var_name="Year",
                  value_name="Value")

df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["Year"]  = pd.to_numeric(df_long["Year"],  errors="coerce")
df_long = df_long.dropna()

# Pivot to 2D matrix: regions × years
df_pivot = df_long.pivot(index="Country Name", columns="Year", values="Value")

# Clustermap:
# - col_cluster=False preserves the original temporal (chronological) order of years
# - Only rows (regions) are hierarchically clustered to reveal similarity
# - method="ward": minimises within-cluster variance — produces compact, well-separated clusters
# - metric="euclidean": measures straight-line distance between region time-series vectors
# - z_score=0: standardise across rows so regions with different absolute levels
#              are compared by their *shape* over time, not their magnitude
g = sns.clustermap(
    df_pivot,
    col_cluster=False,              # Keep years in chronological order
    row_cluster=True,               # Cluster regions by similarity
    cmap="Reds",                  # Diverging: below-average trend = blue, above = red
    linewidths=0.3,
    annot=True,
    fmt=".1f",
    figsize=(20, 6),
    dendrogram_ratio=(0.15, 0.05),
    cbar_pos=(1.02, 0.3, 0.02, 0.4),
    cbar_kws={"label": "Z-score (standardised unemployment)"},
)

g.ax_col_dendrogram.set_title(
    "Youth Unemployment Rates (2000–2025) Regionally",
    fontsize=13, pad=10
)
g.ax_heatmap.set_xlabel("", fontsize=11)
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.tick_params(axis="x", labelrotation=45, labelsize=8)
g.ax_heatmap.tick_params(axis="y", labelrotation=0,  labelsize=9)

plt.savefig("clustermap.png", dpi=150, bbox_inches="tight")
plt.show()
