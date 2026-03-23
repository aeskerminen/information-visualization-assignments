import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
import plotly.express as px
from sklearn.manifold import MDS

from sklearn.preprocessing import StandardScaler, scale


def print_mds_summary(title, mds_model):
    print(f"\n=== {title} ===")
    print(f"Stress (lower is better): {mds_model.stress_:.4f}")

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


X = df_pivot[["GDP_Trillion_USD", "Inflation", "GHG_per_Capita", "Youth_Unemployment"]]

# MDS on non-standardized data using two different random seeds, side by side
seeds = [42, 7]
components_list = []
mds_models = []
for seed in seeds:
    mds_raw = MDS(n_components=2, random_state=seed, n_init=10)
    components_raw = mds_raw.fit_transform(X)
    print_mds_summary(f"MDS without Standardization (seed={seed})", mds_raw)
    components_list.append(components_raw)
    mds_models.append(mds_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (components_raw, seed) in enumerate(zip(components_list, seeds)):
    scatter = axes[i].scatter(
        components_raw[:, 0],
        components_raw[:, 1],
        c=df_pivot["Year"],
        cmap="viridis",
    )
    raw_min = min(components_raw[:, 0].min(), components_raw[:, 1].min())
    raw_max = max(components_raw[:, 0].max(), components_raw[:, 1].max())
    axes[i].set_xlim(raw_min, raw_max)
    axes[i].set_ylim(raw_min, raw_max)
    axes[i].set_title(f"MDS (seed={seed})")
    axes[i].set_xlabel("MDS1")
    axes[i].set_ylabel("MDS2")
    plt.colorbar(scatter, ax=axes[i], label="Year")

plt.tight_layout()
plt.show()