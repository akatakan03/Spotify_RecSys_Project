import polars as pl
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

print("🚀 Kurumsal Seviye Hibrit Öneri Motoru (Sparse Matrix Edition) Başlatılıyor...\n")

# 1. Altyapıyı Yükleme
matrix_path = 'data/processed/sparse_cooccurrence.npz'
mapping_path = 'data/processed/uri_mapping.pkl'
metadata_path = 'data/processed/offline_track_metadata.parquet'
features_path = 'data/processed/audio_features_v2.parquet'

sparse_matrix = load_npz(matrix_path)
with open(mapping_path, 'rb') as f:
    mapping = pickle.load(f)
    uri_to_idx = mapping['uri_to_idx']
    idx_to_uri = mapping['idx_to_uri']

df_metadata = pl.read_parquet(metadata_path).to_pandas()
df_features = pl.read_parquet(features_path).to_pandas()
df_merged = pd.merge(df_features, df_metadata, on='track_uri')

# 2. Ölçeklendirme
scaler = StandardScaler()
feature_cols = ['bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'harmonic_ratio', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5']
df_merged[feature_cols] = scaler.fit_transform(df_merged[feature_cols])

def hibrit_oner(seed_uris, w_audio=0.5, w_co=0.5, top_n=10):
    """
    Hem Ses (Audio) hem de İnsan Davranışı (Co-occurrence) kullanan ana fonksiyon.
    """
    # Mevcut şarkıları indekslere çevir
    seed_indices = [uri_to_idx[uri] for uri in seed_uris if uri in uri_to_idx]
    
    if not seed_indices:
        return "Hata: Tohum şarkılar kütüphanede bulunamadı."

    # A) AUDIO COMPONENT (SES BİLEŞENİ)
    # Tohum şarkıların ortalama vektörünü (User Profile) oluştur
    user_profile = df_merged[df_merged['track_uri'].isin(seed_uris)][feature_cols].mean().values.reshape(1, -1)
    # Tüm kütüphane ile benzerliği ölç
    audio_sims = (cosine_similarity(user_profile, df_merged[feature_cols].values)[0] + 1) / 2
    
    # B) COLLABORATIVE COMPONENT (İNSAN DAVRANIŞI BİLEŞENİ)
    # Seyrek matris üzerinden tohum şarkıların satırlarını topla (Süper hızlı işlem)
    co_scores = np.array(sparse_matrix[seed_indices, :].sum(axis=0)).flatten()
    # Skorları 0-1 arasına normalize et (Eğer veri varsa)
    if co_scores.max() > 0:
        co_scores = co_scores / co_scores.max()
    
    # C) HYBRID INTEGRATION (HİBRİT BİRLEŞTİRME)
    final_scores = (audio_sims * w_audio) + (co_scores * w_co)
    
    # Sonuçları sırala
    results_idx = np.argsort(final_scores)[::-1]
    
    oneriler = []
    for idx in results_idx:
        uri = idx_to_uri[idx]
        if uri not in seed_uris: # Kendini önerme
            oneriler.append({
                'name': df_metadata[df_metadata['track_uri'] == uri]['track_name'].values[0],
                'artist': df_metadata[df_metadata['track_uri'] == uri]['artist_name'].values[0],
                'score': final_scores[idx]
            })
        if len(oneriler) == top_n:
            break
            
    return oneriler

# ÖRNEK KULLANIM (Test)
print("Örnek Test Çalıştırılıyor...")
test_seeds = df_metadata['track_uri'].head(3).tolist() # İlk 3 şarkıyı tohum alalım
sonuclar = hibrit_oner(test_seeds, w_audio=0.4, w_co=0.6)

for i, s in enumerate(sonuclar):
    print(f"{i+1}. {s['name']} - {s['artist']} (Hibrit Skor: {s['score']:.4f})")