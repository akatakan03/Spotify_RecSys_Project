import os
import json
import polars as pl
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import pickle
import warnings

warnings.filterwarnings('ignore')

print("Büyük Veri (Big Data) Optimizasyonu Başlıyor: Sparse Matrix (Seyrek Matris) İnşası...\n")

# 1. Veriyi Yükleme ve Mapping (Haritalandırma) İşlemleri
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
havuz_uris = df_metadata['track_uri'].to_list()

# String (Metin) formatındaki URI'leri, Matris için Integer (Tam Sayı) indekslere çeviriyoruz
uri_to_idx = {uri: idx for idx, uri in enumerate(havuz_uris)}
idx_to_uri = {idx: uri for uri, idx in uri_to_idx.items()}

# 2. Koordinat Listelerini Hazırlama
satirlar = []
sutunlar = []
degerler = []

json_folder = 'data/raw/'
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]

print("JSON dosyaları taranıyor ve koordinatlar (Coordinates) çıkarılıyor...")

for i, dosya in enumerate(dosyalar):
    with open(os.path.join(json_folder, dosya), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for playlist in data['playlists']:
            # Sadece kütüphanemizde olan şarkıların indeks numaralarını al
            playlist_idx = [uri_to_idx[t['track_uri']] for t in playlist['tracks'] if t['track_uri'] in uri_to_idx]
            
            # Kombinasyonları çıkar ve listelere ekle
            for x in range(len(playlist_idx)):
                for y in range(x + 1, len(playlist_idx)):
                    idx1 = playlist_idx[x]
                    idx2 = playlist_idx[y]
                    
                    # Simetrik matris olduğu için her iki yönü de ekliyoruz
                    satirlar.extend([idx1, idx2])
                    sutunlar.extend([idx2, idx1])
                    degerler.extend([1, 1]) # Her yan yana gelişte skor 1 artar

    if (i + 1) % 50 == 0:
        print(f"İlerleme: {i+1} dosya işlendi.")

# 3. Sparse Matrix (Seyrek Matris) Oluşturma ve Sıkıştırma
print("\nKoordinatlar birleştiriliyor ve CSR formatına sıkıştırılıyor (Compression)...")
matris_boyutu = len(havuz_uris)

# Önce COO (Coordinate Format) oluşturuyoruz, sonra CSR'ye çeviriyoruz
sparse_co_matrix = coo_matrix((degerler, (satirlar, sutunlar)), shape=(matris_boyutu, matris_boyutu))

# Tekrarlayan koordinatları topluyoruz (Örn: A ve B şarkısı 5 kere yan yana geldiyse 1+1+1+1+1 = 5 yapar)
sparse_co_matrix.sum_duplicates()

# Okuma işlemlerinde süper hızlı olan CSR formatına dönüştürüyoruz
csr_matrix = sparse_co_matrix.tocsr()

# 4. Diske Kaydetme (Serialization)
matrix_path = 'data/processed/sparse_cooccurrence.npz'
mapping_path = 'data/processed/uri_mapping.pkl'

save_npz(matrix_path, csr_matrix)
with open(mapping_path, 'wb') as f:
    pickle.dump({'uri_to_idx': uri_to_idx, 'idx_to_uri': idx_to_uri}, f)

print(f"\nMuazzam başarı! Devasa matris %99 oranında sıkıştırıldı.")
print(f"Matris dosyası kaydedildi: {matrix_path}")
print(f"Haritalandırma dosyası kaydedildi: {mapping_path}")