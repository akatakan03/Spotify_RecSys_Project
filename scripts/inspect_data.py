import polars as pl

# 1. Metadata Dosyasını İnceleme
print("--- METADATA DOSYASI (Şarkı Bilgileri) ---")
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
print(df_metadata.head(5))
print(f"Toplam Satır (Row) Sayısı: {len(df_metadata)}\n")

# 2. Audio Features Dosyasını İnceleme
print("--- AUDIO FEATURES DOSYASI (Ses Analizleri) ---")
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
print(df_features.head(5))
print(f"Toplam Satır (Row) Sayısı: {len(df_features)}\n")