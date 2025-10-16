import pandas as pd
import locale

# Jalur file CSV
file_path = r'C:\Users\galan\Music\projek_new\lbp.csv'
output_path = r'C:\Users\galan\Music\projek_new\lbp_baru.csv'

# Membaca file CSV dengan delimiter titik koma (;)
df = pd.read_csv(file_path, delimiter=';')

# Menampilkan informasi awal
print("=== INFORMASI AWAL DATAFRAME ===")
print("\nNama-nama kolom header:")
print(df.columns.tolist())

print("\nTipe data setiap kolom sebelum perubahan:")
print(df.dtypes)

# Proses konversi tipe data untuk kolom LBP
lbp_columns = ['LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_9']

# Cek apakah kolom-kolom tersebut ada di dataframe
existing_lbp_cols = [col for col in lbp_columns if col in df.columns]

if existing_lbp_cols:
    print("\n=== MENGUBAH TIPE DATA KOLOM LBP ===")
    
    # Atur locale untuk format angka (gunakan 'de_DE' untuk format Eropa dengan koma desimal)
    locale.setlocale(locale.LC_NUMERIC, 'de_DE' if ',' in df[existing_lbp_cols[0]].iloc[0] else 'en_US')
    
    # Fungsi untuk konversi string ke float dengan format yang benar
    def convert_to_float(value):
        try:
            if isinstance(value, str):
                # Ganti titik pemisah ribuan jika ada
                value = value.replace('.', '').replace(',', '.')
            return float(value)
        except (ValueError, AttributeError):
            return None
    
    # Konversi ke float
    for col in existing_lbp_cols:
        df[col] = df[col].apply(convert_to_float)
        print(f"\n5 sampel pertama kolom {col} setelah konversi:")
        print(df[col].head())
    
    print("\nTipe data setelah perubahan:")
    print(df[existing_lbp_cols].dtypes)
else:
    print("\nTidak ada kolom LBP yang ditemukan untuk dikonversi")

# Menyimpan dataframe ke file baru
df.to_csv(output_path, sep=';', index=False, float_format='%.15g')  # format untuk mencegah notasi ilmiah
print(f"\nDataframe telah disimpan ke {output_path} dengan tipe data yang telah diupdate")

# Informasi tambahan
print("\n=== INFORMASI TAMBAHAN ===")
print("\nJumlah data per kolom:")
print(df.count())

label_column = 'label'  # Ganti dengan nama kolom label yang sesuai
if label_column in df.columns:
    print(f"\nJumlah data per label kelas ({label_column}):")
    print(df[label_column].value_counts())
else:
    print(f"\nKolom label '{label_column}' tidak ditemukan.")

print("\nJumlah missing value per kolom:")
print(df.isnull().sum())