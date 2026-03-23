import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, scale


def print_pca_summary(title, pca_model, feature_names):
    print(f"\n=== {title} ===")
    evr = pca_model.explained_variance_ratio_
    print(
        "Explained variance: "
        f"PC1={evr[0] * 100:.2f}% | PC2={evr[1] * 100:.2f}% | "
        f"Total={evr[0] * 100 + evr[1] * 100:.2f}%"
    )

    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=feature_names,
        columns=["PC1", "PC2"],
    )
    print("\nLoadings (weights per variable):")
    print(loadings.round(4).to_string())

    for pc_name in ["PC1", "PC2"]:
        dominant_feature = loadings[pc_name].abs().idxmax()
        dominant_weight = loadings.loc[dominant_feature, pc_name]
        print(
            f"Dominant variable for {pc_name}: {dominant_feature} "
            f"(loading={dominant_weight:.4f})"
        )

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


# PCA without standardization
pca_raw = PCA(n_components=2)
components_raw = pca_raw.fit_transform(X)

# PCA with standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_std = PCA(n_components=2)
components_std = pca_std.fit_transform(X_scaled)

# Print interpretation details so PC1/PC2 are understandable from output
print_pca_summary("PCA without Standardization", pca_raw, X.columns)
print_pca_summary("PCA with Standardization", pca_std, X.columns)

# Plot both results side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Unstandardized PCA
scatter1 = axes[0].scatter(components_raw[:, 0], components_raw[:, 1], c=df_pivot['Year'], cmap='viridis')
raw_min = min(components_raw[:, 0].min(), components_raw[:, 1].min())
raw_max = max(components_raw[:, 0].max(), components_raw[:, 1].max())
axes[0].set_xlim(raw_min, raw_max)
axes[0].set_ylim(raw_min, raw_max)
axes[0].set_title('PCA without Standardization')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Year')

# Standardized PCA
scatter2 = axes[1].scatter(components_std[:, 0], components_std[:, 1], c=df_pivot['Year'], cmap='viridis')
axes[1].set_title('PCA with Standardization')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
cbar2 = plt.colorbar(scatter2, ax=axes[1], label='Year')

plt.tight_layout()
plt.show()