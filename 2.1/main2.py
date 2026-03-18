import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("youth_unemployment_2000_2025_majorregions.csv")
df.columns = [col.split(" ")[0] if "[" in col else col for col in df.columns]

df_long = df.melt(id_vars=["Country Name"], 
                  var_name="Year", 
                  value_name="Value")

plt.figure(figsize=(16, 5))

df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long = df_long.dropna()

# Pivot to 2D matrix: regions × years
df_pivot = df_long.pivot(index="Country Name", columns="Year", values="Value")
df_pivot = df_pivot.sort_index()

sns.heatmap(
    data=df_pivot,
    cmap="Reds",
    linewidths=0.3,
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "Youth Unemployment Rate (%)"}
)

plt.title("Youth Unemployment Rates (2000-2025) Regionally", fontsize=14)
plt.xlabel("", fontsize=12)
plt.ylabel("", fontsize=12)
plt.xticks(rotation=45)

sns.despine(trim=True)
plt.tight_layout()
plt.show()