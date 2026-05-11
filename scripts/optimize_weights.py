import os
import json
import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import math
import pickle
import warnings

warnings.filterwarnings('ignore')

print("Hyperparameter Tuning (Ağırlık Optimizasyonu) Motoru Başlatılıyor...\n")

# 1. Verileri Yükleme (Load Data)
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
df_merged = df_features.join(df_metadata, on='track_uri', how='inner').to_pandas()

with open('data/processed/co_occurrence.pkl', 'rb') as f:
    co_occurrence = pickle.load(f)

scaler = StandardScaler()
feature_cols = ['bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'harmonic_ratio', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5']
df_merged[feature_cols] = scaler.fit_transform(df_merged[feature_cols])

havuz_uris = set(df_merged['track_uri'].to_list())
tum_ozellikler = df_merged[feature_cols].values

# 2. Çalma Listelerini Hazırlama (Prepare Playlists)
json_folder = 'data/raw/'
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]
uygun_playlistler = []

for dosya in dosyalar:
    with open(os.path.join(json_folder, dosya), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for playlist in data['playlists']:
            playlist_uris = [t['track_uri'] for t in playlist['tracks']]
            ortak_sarkilar = [uri for uri in playlist_uris if uri in havuz_uris]
            if len(ortak_sarkilar) >= 10:
                uygun_playlistler.append({'name': playlist['name'], 'tracks': ortak_sarkilar[:10]})
    if len(uygun_playlistler) >= 50:
        break

# 3. Grid Search (Tüm Kombinasyonları Test Etme)
# 0.0'dan 1.0'a kadar 0.1 adımlarla ağırlıkları oluşturuyoruz.
weight_combinations = [(round(a, 1), round(1.0 - a, 1)) for a in np.arange(0.0, 1.1, 0.1)]

best_ndcg = 0
best_weights = (0, 0)
sonuclar_log = []

print("Tüm olasılıklar taranıyor (Grid Search). Lütfen bekleyin...\n")

for w_audio, w_co in weight_combinations:
    toplam_ndcg = 0
    
    for pl_data in uygun_playlistler:
        kesisen_sarkilar = pl_data['tracks']
        seed_tracks = kesisen_sarkilar[:8]
        target_tracks = kesisen_sarkilar[8:]
        
        seed_df = df_merged[df_merged['track_uri'].isin(seed_tracks)]
        user_profile = seed_df[feature_cols].mean().values.reshape(1, -1)
        audio_sim_skorlari = (cosine_similarity(user_profile, tum_ozellikler)[0] + 1) / 2
        
        hibrit_skorlar = []
        for idx in range(len(df_merged)):
            uri = df_merged.iloc[idx]['track_uri']
            if uri in seed_tracks:
                continue
                
            co_skor = 0
            for seed in seed_tracks:
                co_skor += co_occurrence.get(seed, {}).get(uri, 0)
            
            audio_skor = audio_sim_skorlari[idx]
            
            # Dinamik Ağırlıklandırma (Dynamic Weighting)
            if co_skor > 0:
                final_skor = (audio_skor * w_audio) + (co_skor * w_co)
            else:
                final_skor = (audio_skor * 1.0)
                
            hibrit_skorlar.append((uri, final_skor))
            
        hibrit_skorlar.sort(key=lambda x: x[1], reverse=True)
        top_k_oneriler = [x[0] for x in hibrit_skorlar[:50]]
        
        dcg = sum([1 / math.log2(i + 1 + 1) for i, uri in enumerate(top_k_oneriler) if uri in target_tracks])
        idcg = sum([1 / math.log2(i + 1 + 1) for i in range(len(target_tracks))])
        
        ndcg = dcg / idcg if idcg > 0 else 0
        toplam_ndcg += ndcg
        
    ortalama_ndcg = toplam_ndcg / len(uygun_playlistler)
    sonuclar_log.append(f"Ses %{int(w_audio*100)} / İnsan %{int(w_co*100)} -> NDCG: {ortalama_ndcg:.4f}")
    
    # En iyi skoru güncelleme (Update Best Score)
    if ortalama_ndcg > best_ndcg:
        best_ndcg = ortalama_ndcg
        best_weights = (w_audio, w_co)

# 4. Sonuçları Raporlama
print("-" * 50)
print("📊 BİLİMSEL TEST SONUÇLARI (Tüm Kombinasyonlar):")
for log in sonuclar_log:
    print(log)

print("-" * 50)
print(f"🏆 MATEMATİKSEL OLARAK KANITLANMIŞ EN İYİ KOMBİNASYON:")
print(f"🔊 Audio Weight (Ses): %{int(best_weights[0]*100)}")
print(f"👥 Co-occurrence Weight (İnsan): %{int(best_weights[1]*100)}")
print(f"🥇 Maksimum NDCG Skoru: {best_ndcg:.4f}")
print("-" * 50)