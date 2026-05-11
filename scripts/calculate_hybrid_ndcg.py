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

print("Gelişmiş Hibrit (Audio + Co-occurrence) NDCG Testi Başlıyor...\n")

# 1. Verileri ve İlişki Ağını Yükleme
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet')
df_features = pl.read_parquet('data/processed/audio_features_v2.parquet')
df_merged = df_features.join(df_metadata, on='track_uri', how='inner').to_pandas()

with open('data/processed/co_occurrence.pkl', 'rb') as f:
    co_occurrence = pickle.load(f)

# 2. Ölçeklendirme (Standardization)
scaler = StandardScaler()
feature_cols = ['bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'harmonic_ratio', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5']
df_merged[feature_cols] = scaler.fit_transform(df_merged[feature_cols])

havuz_uris = set(df_merged['track_uri'].to_list())
tum_ozellikler = df_merged[feature_cols].values

# 3. Uygun Çalma Listelerini Bulma
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

toplam_ndcg = 0
toplam_r_precision = 0

# 4. Sınav: Hibrit Skorlama (Hybrid Scoring)
for pl_data in uygun_playlistler:
    kesisen_sarkilar = pl_data['tracks']
    seed_tracks = kesisen_sarkilar[:8]
    target_tracks = kesisen_sarkilar[8:]
    
    # A) Audio Profilini Çıkarma (Content-Based)
    seed_df = df_merged[df_merged['track_uri'].isin(seed_tracks)]
    user_profile = seed_df[feature_cols].mean().values.reshape(1, -1)
    audio_sim_skorlari = cosine_similarity(user_profile, tum_ozellikler)[0]
    
    # Skorları 0-1 arasına çekme (Normalization)
    audio_sim_skorlari = (audio_sim_skorlari + 1) / 2
    
    hibrit_skorlar = []
    
    # B) Tüm Şarkılar İçin Co-occurrence Skoru Hesaplama (Collaborative Filtering)
    for idx in range(len(df_merged)):
        uri = df_merged.iloc[idx]['track_uri']
        if uri in seed_tracks:
            continue # Zaten tohumda olanları atla
            
        # Bu aday şarkının, 8 tohum şarkıyla toplam yan yana gelme sayısını bul
        co_skor = 0
        for seed in seed_tracks:
            co_skor += co_occurrence.get(seed, {}).get(uri, 0)
        
        # C) Hybrid Equation (Hibrit Denklem)
        audio_skor = audio_sim_skorlari[idx]
        
        if co_skor > 0:
            final_skor = (audio_skor * 0.1) + (co_skor * 0.9)
        else:
            final_skor = (audio_skor * 1.0)
            
        hibrit_skorlar.append((uri, final_skor))
        
    # 5. Sıralama ve Değerlendirme (Ranking & Evaluation)
    hibrit_skorlar.sort(key=lambda x: x[1], reverse=True)
    top_k_oneriler = [x[0] for x in hibrit_skorlar[:50]]
    
    # R-Precision
    dogru_tahminler = set(top_k_oneriler).intersection(set(target_tracks))
    r_prec = len(dogru_tahminler) / len(target_tracks)
    toplam_r_precision += r_prec
    
    # NDCG
    dcg = 0
    idcg = 0
    for i, uri in enumerate(top_k_oneriler):
        if uri in target_tracks:
            dcg += 1 / math.log2(i + 1 + 1)
    for i in range(len(target_tracks)):
        idcg += 1 / math.log2(i + 1 + 1)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    toplam_ndcg += ndcg

ortalama_r_precision = toplam_r_precision / len(uygun_playlistler)
ortalama_ndcg = toplam_ndcg / len(uygun_playlistler)

print("-" * 50)
print(f"🎯 HİBRİT ORTALAMA R-Precision Skoru: {ortalama_r_precision:.4f}")
print(f"🥇 HİBRİT ORTALAMA NDCG Skoru: {ortalama_ndcg:.4f}")
print("-" * 50)