import pandas as pd

# Jalur file CSV
file_path1 = r'C:\Users\galan\Music\projek_new\lbp_baru.csv'
file_path2 = r'C:\Users\galan\Music\projek_new\fch.csv'

# Membaca file CSV dengan delimiter titik koma (;)
df1 = pd.read_csv(file_path1, delimiter=';')
df2 = pd.read_csv(file_path2, delimiter=';')

# Menggabungkan kedua DataFrame berdasarkan kolom 'filename' dan 'label'
merged_df = pd.merge(df1, df2, on=['filename', 'label'], how='inner')

# Membersihkan data: Ganti koma dengan titik dan konversi ke float
for col in merged_df.columns:
    if merged_df[col].dtype == object:  # Jika kolom bertipe object (string)
        try:
            # Ganti koma dengan titik dan konversi ke float
            merged_df[col] = merged_df[col].str.replace(',', '.').astype(float)
        except (ValueError, AttributeError):
            # Jika kolom tidak bisa diubah ke float (misalnya, kolom 'filename' atau 'label'), lewati
            continue

# Menampilkan hasil penggabungan dan pembersihan
print("Data setelah digabungkan dan dibersihkan:")
print(merged_df)

# Menyimpan hasil penggabungan ke file CSV baru (opsional)
output_path = r'C:\Users\galan\Music\projek_new\gabung.csv'
merged_df.to_csv(output_path, index=False, sep=';')
print(f"\nData gabungan telah disimpan di: {output_path}")