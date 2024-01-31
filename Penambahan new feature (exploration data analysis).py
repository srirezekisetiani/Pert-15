# Import library yang diperlukan
import pandas as pd
import numpy as np

# Memuat dataset
dataset = pd.read_csv('data.csv')

# Eksplorasi Data Analysis
# Contoh 1: Menambahkan kolom baru yang merupakan jumlah dari dua kolom lainnya
dataset['new_feature'] = dataset['feature1'] + dataset['feature2']

# Contoh 2: Menghitung rata-rata dari suatu kolom
mean_value = dataset['feature3'].mean()

# Contoh 3: Mengganti nilai yang hilang (missing values) dengan nilai rata-rata
dataset['feature4'] = dataset['feature4'].fillna(mean_value)

# Contoh 4: Mengubah tipe data kolom menjadi kategori
dataset['category'] = dataset['category'].astype('category')

# Contoh 5: Menggunakan teknik one-hot encoding untuk kolom kategori
one_hot_encoded = pd.get_dummies(dataset['category'])

# Contoh 6: Menggabungkan dataset dengan dataset lain berdasarkan kolom tertentu
other_dataset = pd.read_csv('other_data.csv')
merged_dataset = pd.merge(dataset, other_dataset, on='id')

# Contoh 7: Melakukan analisis statistik sederhana
min_value = dataset['feature1'].min()
max_value = dataset['feature1'].max()
std_dev = dataset['feature1'].std()

# Contoh 8: Menyimpan dataset yang telah diubah ke dalam file CSV
dataset.to_csv('modified_data.csv', index=False)

# Contoh 9: Menampilkan ringkasan statistik dari dataset
summary_stats = dataset.describe()

# Contoh 10: Menampilkan visualisasi data menggunakan library seperti matplotlib atau seaborn
import matplotlib.pyplot as plt
plt.scatter(dataset['feature1'], dataset['feature2'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot')
plt.show()
