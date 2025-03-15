import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Veri setini yükledim
file_path = "Sleep_health_and_lifestyle_dataset.csv"  # Dosyanın adını ve yolunu doğru yazmaya özen gösterdim
df = pd.read_csv(file_path)

# İlk birkaç satırı görüntüleyerek veri setimi kontrol ettim
print(df.head())

# Genel bilgi alarak veri setimin konusunu inceledim
print(df.info())

# Tanımlayıcı istatistikler
print(df.describe())

# Eksik verileri kontrol ettim
print(df.isnull().sum())

# Eksik verileri doldurmasını veya silmesini(opsiyonel) istedim
df = df.dropna()  # axis=1 parametresi kullanarak da yapabilirim bu işlemi

# Veriyi görselleştirdim
plt.figure(figsize=(10, 5))
sns.histplot(df["Sleep Duration"], bins=30, kde=True)
plt.title("Uyku Süresi Dağılımı")
plt.show()

# Kategorik değişkenleri analiz etme
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=df)
plt.title("Cinsiyete Göre Dağılım")
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()

# Standardizasyon (Z-score normalization)
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['number']).columns  # Sayısal sütunları otomatik seç
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Min-Max Normalizasyonu
min_max_scaler = MinMaxScaler()
df[numeric_cols] = min_max_scaler.fit_transform(df[numeric_cols])

print(df.head())


# Uyku süresi dağılımı
plt.figure(figsize=(8, 5))
sns.histplot(df['Sleep Duration'], bins=30, kde=True, color='blue')
plt.title('Uyku Süresi Dağılımı')
plt.xlabel('Uyku Süresi (Standartlaştırılmış)')
plt.ylabel('Frekans')
plt.show()

# VERİ GOSELLESTIRME
# Yaşa göre Uyku Süresi Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Age'], y=df['Sleep Duration'], hue=df['Gender'])
plt.title('Yaş ve Uyku Süresi İlişkisi')
plt.xlabel('Yaş')
plt.ylabel('Uyku Süresi')
plt.show()

# GroupBy ile Gruplama İşlemleri
# Cinsiyete göre ortalama uyku süresi
grouped_gender = df.groupby('Gender')['Sleep Duration'].mean()
print(grouped_gender)

# Stres seviyesine göre ortalama kalp hızı
grouped_stress = df.groupby('Stress Level')['Heart Rate'].mean()
print(grouped_stress)

# Select, Where ve Query ile Filtreleme
# 30 yaş üstü ve 6 saatten az uyuyanları seçelim
filtered_df = df[(df['Age'] > 30) & (df['Sleep Duration'] < -0.5)]  # Z-score kullanıldığı için -0.5 threshold seçildi
print(filtered_df.head())

# Query ile stres seviyesi yüksek olanlar
high_stress_df = df.query("`Stress Level` > 0.8")
print(high_stress_df.head())


correlation = df[['Sleep Duration', 'Stress Level']].corr()
print("Uyku Süresi ve Stres Seviyesi Korelasyonu:\n", correlation)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Uyku Süresi ve Stres Seviyesi Korelasyon Matrisi")
plt.show()

# Yeni bir sahte veri seti oluşturdum
df2 = pd.DataFrame({
    'Person ID': df.index,
    'BMI': np.random.uniform(18, 30, size=len(df))  # Rastgele BMI değerleri
})

# merge ve join ile veri birleştirme yaptım
merged_df = df.merge(df2, left_index=True, right_on='Person ID', how='inner')
print(merged_df.head())

# Eksik verileri kontrol ettim
print(df.isnull().sum())

# Eksik verileri ortalama ile doldurdum
df.fillna(df.mean(), inplace=True)

# One-Hot Encoding
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

print(df.head())
