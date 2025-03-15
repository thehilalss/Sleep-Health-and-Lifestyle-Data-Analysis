import numpy as np
import pandas as pd

# endregion

# region IMDB - Select - Filter
df = pd.read_csv(
    filepath_or_buffer='data/imdb.csv',
    encoding='utf-8'
)
print(df.head().to_string())

print(df.shape)

print(df.describe())

# Movie_Title ilk 20 satırı listeleyin
# Path 1
print(df[['Movie_Title']].head(20))
# Path 2
print(df[['Movie_Title']][0:20])
# Path 3
print(df.loc[:20, 'Movie_Title'])

# Rating 7.0 dan büyük olan filmlerin
# select --> Movie_Title, Rating, YR_Released
# 20 ile 50 arasındaki filmler gelsin
# sort edelim

print(
    df[df['Rating'] > 7.0]
    [['Movie_Title', 'Rating', 'YR_Released']]
    [20:50].sort_values(by='Rating', ascending=False)
)

# YR_Released 2014 ile 2019 arasında olan
# Movie_Title, Rating, YR_Released
print(
    df[
        (df['YR_Released'] >= 2014) & (df['YR_Released'] <= 2019)
        ]
    [['Movie_Title', 'Rating', 'YR_Released']]
)

print(
    df[
        df['YR_Released'].between(2014, 2019)
    ]
    [['Movie_Title', 'Rating', 'YR_Released']]
)

# Num_Reviews 100.00'den büyüjk yada Rating 8 ile 9 arsında olan
# Movie_Title, Rating, YR_Released
print(
    df[
        (df['Num_Reviews'] >= 100) | ((df['Rating'] > 8.0) & (df['Rating'] < 9.0))
        ]
    [['Movie_Title', 'Rating', 'YR_Released']].sort_values(by='Num_Reviews', ascending=False)
)
# endregion


# region NBA - Group By- Select - Filter
df = pd.read_csv(filepath_or_buffer='data/nba.csv')

print(df.head(20).to_string())
print(df.shape)

# En yüksek maaşlı oyuncuyu bulalım
print(
    df[
        df['Salary'] == df['Salary'].max()
        ]
    [['Name', 'Team', 'Age', 'Salary']]
)

# group by
print(
    df.groupby(by='Team')['Salary'].sum()
)

# Kaç farklı takım var
print(df['Team'].nunique())

# yaşları 20 ile 35 arasına olan adı, takımı, bilgilerini listeleyelim
print(
    df[
        df['Age'].between(20, 35)
    ]
    [['Name', 'Team', 'Age']].sort_values(by='Age', ascending=False).groupby('Team')['Age'].mean()
)

print(df.groupby(by='Team')['Age'].mean())


# Custom Function
def str_find(name: str):
    if name.__contains__('and'):
        return True
    else:
        return False


print(df['Name'].apply(str_find))

print(df['Name'].apply(lambda x: x.str_find))

# endregion


# region youtube


print(df.head().to_string())
print(df.shape)
print(df.columns)

# en çok görüntülenme alan il 10 videonun adını, view
print(
    df.groupby('title')[['views']].sum().sort_values(by='views', ascending=False).head(10)
)

# catgory_id sütununa göre grouplayalım like sayısını saptayalım
print(
    df.groupby('category_id')[['likes']].sum().sort_values(by='likes', ascending=False).head(10)
)

# hangi channel_title  ne kadar comment_count
print(
    df.groupby('channel_title')[['comment_count']].sum().sort_values(by='comment_count', ascending=False).head(10)
)


# hangi videonun kaç tane tag var
def calcualte_tag_count(tag: str):
    return len(tag.split('|'))


# Path I
df['tag_count'] = df['tags'].apply(calcualte_tag_count)
# path II
df['tag_count'] = df['tags'].apply(lambda x: x.calcualte_tag_count)

print(df.head().to_string())

print(
    df.groupby('title')[['tag_count']].sum().sort_values(by='tag_count', ascending=False).head(10)
)


def like_dislike_avg(data_set: pd.DataFrame) -> list:
    like_list = list(data_set['likes'])
    dislike_list = list(data_set['dislikes'])

    comb_list = list(zip(like_list, dislike_list))
    avg_like = list()
    for like, dislike in comb_list:
        if like + dislike == 0:
            avg_like.append(0)
        else:
            avg_like.append(like / (like + dislike))

    return avg_like


df['like_dislike_avg'] = like_dislike_avg(data_set=df)

print(
    df.loc[:, ['title', 'likes', 'dislikes', 'like_dislike_avg']].sort_values(by='like_dislike_avg',
                                                                              ascending=False).head(10)
)
# endregion

# ------------------------------------------------------------ 7. EĞİTİM DE YAZILAN KODLAR ------------------------------------------------------------

# region Auto.csv
df = pd.read_csv(filepath_or_buffer='data/auto.csv')

df.columns = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
              "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight",
              "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
              "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

print(df.head().to_string())
print(df.shape)

# Handling Missing Values
# Step I
# Eksik verilerin saptanması
df.replace(
    to_replace='?',
    value=np.nan,
    inplace=True
)

missing_values_df = df.isnull()
print(missing_values_df.head().to_string())

print(missing_values_df.columns.values.tolist())

for item in missing_values_df.columns.values.tolist():
    print(f'Column Name: {item}\nMissing Values Count: {missing_values_df[item].value_counts()}')

# Missing Values Handle
# 1. Ortalama Değer saptayıp eksik veriler yerine yazdırma
df['normalized-losses'] = df['normalized-losses'].replace(to_replace=np.nan,
                                                          value=df['normalized-losses'].astype(float).mean())
print(df['normalized-losses'])
# 2. Frekans aralığını saptayarak eksik verilerin yerine yazdırma
df['num-of-doors'] = df['num-of-doors'].replace(to_replace=np.nan, value=df['num-of-doors'].value_counts().idxmax())
print(df['num-of-doors'])

# Veri Standardizasyonu
df['city_l/km'] = 235 / df['city-mpg']
df['highway_l/km'] = 235 / df['highway-mpg']
print(df[['city_l/km', 'highway_l/km']])

# Normalizasyon
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()

print(df[['length', 'width', 'height']])

# Dummy Variable
dummy_variable_df = pd.get_dummies(df['fuel-type'], dtype=float)
print(dummy_variable_df)

df.to_csv('data/clean_data.csv')
# endregion

# region Canada
import matplotlib.pyplot as plt

df_can = pd.read_excel(
    io='data/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2
)

# print(df_can.head().to_string())

# Sütun isimlerini değiştirme
df_can.rename(
    columns={
        'OdName': 'Country',
        'AreaName': 'Continent',
        'RegName': 'Region'
    },
    inplace=True
)
# print(df_can.head().to_string())

# çalışmada ihtiyaç duyulmayan gereksiz sütunlarda kurtulalım
df_can.drop(
    columns=['Type', 'Coverage', 'AREA', 'REG', 'DEV', 'DevName'],
    axis=1,
    inplace=True
)
# print(df_can.head().to_string())

# sütun başlıklarının tipine
for column in df_can.columns:
    print(type(column))

# göçmen sayılarını tutan sütunların başlıklarının tipini int to str yapalım
df_can.columns = list(map(str, df_can.columns))

for column in df_can.columns:
    print(type(column))

# veri setindeki country sütunu index olarak ayarlayalım ki nokta atışı bir ülkenin verisini çekebilellim
df_can.set_index(keys='Country', inplace=True)
# print(df_can.head().to_string())

# yıl yıl göçmen sayılarını toplayarak total isimli sütun yaratıp içine yazalım
df_can['Total'] = df_can.sum(axis=1, numeric_only=True)
print(df_can.head().to_string())

# veri setinde ki yıllar ile bire bir aynı değerlere sahip bir liste oluşturalım
years = list(map(str, range(1980, 2014)))

# en çok göç vermiş top 5 country saptayalım area grafiğinde gösterelim
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_top_5_country = df_can.head()
print(df_top_5_country.to_string())

df_top_5_country = df_top_5_country[years].transpose()
print(df_top_5_country.to_string())

df_top_5_country.plot(
    kind='area',
    figsize=(10, 7),
    stacked=True,
)

plt.title(label='Immigrant of Top 5 Countries', color='r')
plt.xlabel(xlabel='Years', c='r')
plt.ylabel(ylabel='Number of Immigrant', c='r')
plt.grid()
plt.show()

# histogram grafiği yapalım
# 2013 yılında göçmen vemiş ülkeler
count, bin_edges = np.histogram(df_can['2013'])
print(count)
print(bin_edges)

df_can['2013'].plot(
    kind='hist',
    figsize=(10, 7),
    xticks=bin_edges,
    color='b'
)

plt.title(label='Immigrant to 2013', color='r')
plt.xlabel(xlabel='Number of Immigrant', c='r')
plt.ylabel(ylabel='Number of Country', c='r')
plt.grid()
plt.show()

# baltık ülkelerinin 1980 - 2014
df_baltic_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
print(df_baltic_countries.head().to_string())

count, bin_edges = np.histogram(df_baltic_countries, bins=15)
print(count)
print(bin_edges)

df_baltic_countries.plot(
    kind='hist',
    figsize=(10, 7),
    xticks=bin_edges,
    color=['coral', 'darkblue', 'green'],
    stacked=False,
    alpha=0.25
)

plt.title(label='Immigrant to 2013', color='r')
plt.xlabel(xlabel='Years', c='r')
plt.ylabel(ylabel='Number of Country', c='r')
plt.grid()
plt.show()

# Iceland göçmen harketliliğini çubuk grafikte göserelim
df_iceland = df_can.loc['Iceland', years]

df_iceland.plot(
    kind='bar',
    figsize=(10, 7)
)

plt.title(label='Iceland Immigrant from 1980 to 2014', color='r')
plt.xlabel(xlabel='Years', c='r')
plt.ylabel(ylabel='Number of Immigrant', c='r')
plt.grid()
plt.annotate('',  # s: str. will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

# # Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )

plt.show()

df_continets = df_can.groupby(by='Continent').sum()

df_continets['Total'].plot(
    kind='pie',
    figsize=(10, 7),
    startangle=90,
    autopct='%1.1f%%',
    labels=None,
    shadow=True,
    pctdistance=1.1,
    explode=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
)
plt.axis('equal')
plt.legend(
    labels=df_continets.index,
    prop={
        'size': 8
    }
)
plt.show()
# endregion
