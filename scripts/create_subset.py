import polars as pl

# 1. 2.2 Milyonluk ana veri setimizi okuyoruz
print("Ana veri seti yükleniyor...")
df_master = pl.read_parquet('data/master_tracks.parquet')

# 2. Rastgele 100.000 şarkı seçiyoruz
# seed=42 parametresi, kod her çalıştığında "aynı" rastgele şarkıları seçmesini sağlar.
# Bu sayede deneylerimiz tutarlı ve tekrarlanabilir (reproducible) olur.
df_subset = df_master.sample(n=100000, seed=42)

# 3. Bu yeni küçük listeyi ayrı bir dosya olarak kaydediyoruz
output_path = 'data/subset_100k_tracks.parquet'
df_subset.write_parquet(output_path)

print(f"İşlem tamam! Seçilen şarkı sayısı: {len(df_subset)}")
print(f"Dosya şuraya kaydedildi: {output_path}")