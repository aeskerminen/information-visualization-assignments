import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# t-SNE on non-standardized data with two different random seeds
# for each of three perplexity values (6 runs in one combined figure).
seeds = [42, 7]
perplexities = [5, 10, 15]

fig, axes = plt.subplots(len(perplexities), len(seeds), figsize=(12, 14), constrained_layout=True)

for row, perplexity in enumerate(perplexities):
    for col, seed in enumerate(seeds):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            init="random",
            learning_rate="auto",
            max_iter=1000,
        )

        components = tsne.fit_transform(X)

        ax = axes[row, col]
        scatter = ax.scatter(
            components[:, 0],
            components[:, 1],
            c=df_pivot["Year"],
            cmap="viridis",
            s=60,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_title(f"t-SNE (perplexity={perplexity}, seed={seed})")
        ax.set_xlabel("t-SNE1")
        ax.set_ylabel("t-SNE2")

        print(
            f"Completed perplexity={perplexity} | seed={seed} "
            f"| KL divergence={tsne.kl_divergence_:.4f}"
        )

fig.suptitle("t-SNE on Non-Standardized SDG 8 Data (3 Perplexities x 2 Seeds)", fontsize=14)
fig.colorbar(scatter, ax=axes, label="Year", fraction=0.02, pad=0.02)
plt.savefig("tsne_combined_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved tsne_combined_grid.png")