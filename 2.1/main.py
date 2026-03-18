import json
import pandas as pd
import plotly.express as px

countries = json.load(open("./globe.geo.json"))
df = pd.read_csv("youth_unemployment_globally_2025.csv",
                   na_values=[".."])

midpoint = df["unemployment"].median()

fig = px.choropleth(df, geojson=countries, locations='Country Code',
                           featureidkey="properties.iso_a3",
                           color='unemployment',
                           color_continuous_scale="reds",
                           color_continuous_midpoint=midpoint,
                           range_color=(0, 80),
                           scope="world",
                           labels={'unemployment':'Youth unemployment rate (%)'}
                          )


#fig = px.choropleth(df, geojson=countries, locations='Country Code',
 #                          featureidkey="properties.iso_a3",
  #                         color='unemployment',
    #                       color_continuous_scale="rdylgn_r",
   #                        color_continuous_midpoint=midpoint,
     #                      range_color=(0, 80),
       #                    scope="world",
      #                     labels={'unemployment':'Youth unemployment rate (%)'}
        #                  )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()