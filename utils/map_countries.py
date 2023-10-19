import pandas as pd
import geopandas as gpd
import country_converter as coco
import matplotlib.pyplot as plt

map_file = "../data/country_maps"

#we delete canada (the country that is the most flagged

list_countries = [
    "China", "India", "Nigeria", "Haiti", "Pakistan", "United States of America", "Sri Lanka",
    "Mexico", "Iran", "Brazil", "Colombia", "Vietnam", "Ethiopia",
    "Turkey", "Lebanon", "Somalia", "Ghana", "Hungary", "Jamaica", "Bangladesh",
    "Philippines", "Morocco", "Albania", "Algeria", "Hong Kong S.A.R.", "Afghanistan",
    "Ukraine", "Egypt", "Kenya", "Guyana",
    ]


list_figures = [32336, 21765, 18895, 9917, 8340, 10554, 5987, 5868, 5328, 4118, 3900, 3684, 3347,
                3249, 3174, 3121, 3100, 2888, 2732, 2671, 2513, 2413, 1969, 1806 ,1755,
                1579, 1527, 1512, 1497, 1483, 1389]

data_tuples = list(zip(list_countries,list_figures))
df = pd.DataFrame(data_tuples, columns=['Countries','Occurences'])
print(df.head())

geo_df = gpd.read_file(map_file)[['ADMIN', 'ADM0_A3', 'geometry']]
geo_df.columns = ['country', 'country_code', 'geometry']
print(geo_df.head(3))

geo_df = geo_df.drop(geo_df.loc[geo_df['country'] == 'Antarctica'].index)
iso3_codes = geo_df['country'].to_list()
iso2_codes_list = coco.convert(names=iso3_codes, to='ISO2', not_found='NULL')
geo_df['iso2_code'] = iso2_codes_list
# some countries for which the converter could not find a country code
geo_df = geo_df.drop(geo_df.loc[geo_df['iso2_code'] == 'NULL'].index)


merged_df = pd.merge(left=geo_df, right=df, how='left', left_on='country', right_on='Countries')

merged_df.drop("Countries", inplace=True, axis=1)
merged_df["Occurences"].fillna(0, inplace=True)
print(merged_df.columns)
print(merged_df)

col = "Occurences"
vmin = merged_df[col].min()
vmax = merged_df[col].max()
cmap = 'OrRd'
map = geo_df.plot(edgecolor='grey', linewidth=1)
fig, ax = plt.subplots(1, figsize=(20, 15))
ax.axis('off')
merged_df.plot(column=col, edgecolor='grey', linewidth=1, ax=ax, cmap=cmap)

sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
sm._A = []
cbaxes = fig.add_axes([0.15, 0.30, 0.01, 0.2])
cbar = fig.colorbar(sm, cax=cbaxes)

# save
save_file_name = "../data/map.pdf"
fig.savefig(save_file_name)