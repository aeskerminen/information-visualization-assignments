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

plt.figure(figsize=(12, 7))

df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
df_long["Year"] = pd.to_numeric(df_long["Year"])
df_long = df_long.dropna()


df_long['Era'] = pd.cut(df_long['Year'], 
                        bins=[2000, 2007, 2019, 2025], 
                        labels=['Pre-2008 (2000-2007)', 'Post 2008 (2008-2019)', 'Post CoViD (2020-2025)'])

sns.boxplot(data=df_long, x="Country Name", y="Value", hue="Era")

plt.title("How Youth Unemployment Shifted Across the 2008 Financial Crisis and CoViD-19 Pandemic", fontsize=14)
plt.ylabel("Unemployment Rate (%)")
plt.show()