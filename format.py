import pandas as pd

# Jalur file CSV
file_path = r'C:\Users\galan\Music\src\gabung.csv'

# Membaca file CSV dengan delimiter titik koma (;)
df = pd.read_csv(file_path, delimiter=';')

# Menampilkan nama-nama kolom header
print("Nama-nama kolom header:")
print(df.columns.tolist())

# Menampilkan jumlah data per kolom
print("\nJumlah data per kolom:")
print(df.count())

# Menampilkan tipe data setiap kolom
print("\nTipe data setiap kolom:")
print(df.dtypes)

# Menampilkan jumlah data per label kelas (asumsi kolom label kelas adalah 'label')
label_column = 'label'  # Ganti dengan nama kolom label yang sesuai
if label_column in df.columns:
    print(f"\nJumlah data per label kelas ({label_column}):")
    print(df[label_column].value_counts())
else:
    print(f"\nKolom label '{label_column}' tidak ditemukan.")

# Mengecek missing values di setiap kolom
print("\nJumlah missing value per kolom:")
print(df.isnull().sum())

