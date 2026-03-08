import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and Melt
df = pd.read_csv("sdg8_1.2.csv")
df.columns = [col.split(" ")[0] if "[" in col else col for col in df.columns]

# Include all metadata in id_vars to avoid the "Series Name" error
df_long = df.melt(id_vars=["Country Name", "Series Name"], 
                  var_name="Year", 
                  value_name="Value")

# 2. Clean types
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long = df_long.dropna()
# 1. Pivot the data to get 4 distinct columns
df_pivot = df_long.pivot(index="Year", columns="Series Name", values="Value").reset_index()

# 2. Rename columns to shorter names for the plot axes
# Ensure these match the exact names in your CSV
df_pivot.columns = ["Year", "GHG_per_Capita", "GDP_Growth", "Inflation", "Unemployment"]

# 3. Create a Categorical Year column (Fixes the diagonal disappearance)
df_pivot["Year_Group"] = df_pivot["Year"].astype(str)

# 4. The High-Dimensional Pairplot
sns.set_theme(style="ticks")

g = sns.pairplot(
    df_pivot, 
    vars=["GHG_per_Capita", "GDP_Growth", "Inflation", "Unemployment"],
    hue="Year_Group", 
    diag_kind="kde",  # Restores the distribution humps on the diagonal
    corner=True,      # Removes redundant upper triangle
    palette="viridis" # Color grades from 2000 (dark) to 2025 (yellow)
)

g.figure.suptitle("Pairwise Analysis of Global SDG 8 Indicators", y=1.02)
plt.show()