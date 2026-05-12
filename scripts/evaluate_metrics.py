import os
import json
import numpy as np
import time
import warnings
from recommender_pipeline import SpotifyRecommenderPipeline

warnings.filterwarnings('ignore')

def get_metrics(recommended_uris, truth_uris, k=10):
    """ Precision@K ve nDCG@K hesaplayan yardımcı fonksiyon """
    recommended_k = recommended_uris[:k]
    
    # Precision@K
    hits = len(set(recommended_k) & set(truth_uris))
    precision = hits / k if k > 0 else 0.0
    
    # nDCG@K
    dcg = 0.0
    for i, uri in enumerate(recommended_k):
        if uri in truth_uris:
            dcg += 1.0 / np.log2(i + 2) 
            
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_uris), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return precision, ndcg

print("🎓 Akademik Değerlendirme (Evaluation) Başlatılıyor...\n")

# 1. Pipeline Yükle
pipeline = SpotifyRecommenderPipeline()

# 2. Test için çalma listelerini toplayalım
json_folder = 'data/raw/'
dosyalar = [f for f in os.listdir(json_folder) if f.endswith('.json')]

test_playlists = []
with open(os.path.join(json_folder, dosyalar[0]), 'r', encoding='utf-8') as f:
    data = json.load(f)
    for pl in data['playlists']:
        if len(pl['tracks']) >= 15:
            test_playlists.append([t['track_uri'] for t in pl['tracks']])
        
        if len(test_playlists) == 10000:
            break

print(f"\nToplam {len(test_playlists)} adet çalma listesi taranıyor...")
print("Sistem, bilgisayarımızdaki 5000 şarkı evrenine uyan listeleri filtreliyor...\n")

# YANLIŞLIKLA SİLİNEN KISIM BURASIYDI: Skorları tutacağımız Dictionary (Sözlük)
scores = {
    'Audio_Only': {'prec': [], 'ndcg': [], 'times': []},
    'Collab_Only': {'prec': [], 'ndcg': [], 'times': []},
    'Hybrid': {'prec': [], 'ndcg': [], 'times': []}
}

# 3. Sınavı Başlat (KAPALI DÜNYA TESTİ)
gecerli_test_sayisi = 0

for i, playlist in enumerate(test_playlists):
    bilinen_sarkilar = [uri for uri in playlist if uri in pipeline.uri_to_idx]
    
    if len(bilinen_sarkilar) < 10:
        continue
        
    gecerli_test_sayisi += 1
    
    seed_tracks = bilinen_sarkilar[:5]
    truth_tracks = bilinen_sarkilar[5:]
    
    # --- MODEL 1: Sadece Ses (Audio Only) ---
    start_time = time.time()
    res_audio = pipeline.recommend_hybrid(seed_tracks, w_audio=1.0, w_co=0.0, top_n=10)
    scores['Audio_Only']['times'].append((time.time() - start_time) * 1000)
    p, n = get_metrics([r['uri'] for r in res_audio], truth_tracks)
    scores['Audio_Only']['prec'].append(p)
    scores['Audio_Only']['ndcg'].append(n)

    # --- MODEL 2: Sadece İşbirlikçi (Collab Only) ---
    start_time = time.time()
    res_collab = pipeline.recommend_hybrid(seed_tracks, w_audio=0.0, w_co=1.0, top_n=10)
    scores['Collab_Only']['times'].append((time.time() - start_time) * 1000)
    p, n = get_metrics([r['uri'] for r in res_collab], truth_tracks)
    scores['Collab_Only']['prec'].append(p)
    scores['Collab_Only']['ndcg'].append(n)

    # --- MODEL 3: HİBRİT (Senin Modelin) ---
    start_time = time.time()
    res_hybrid = pipeline.recommend_hybrid(seed_tracks, w_audio=0.2, w_co=0.8, top_n=10)
    scores['Hybrid']['times'].append((time.time() - start_time) * 1000)
    p, n = get_metrics([r['uri'] for r in res_hybrid], truth_tracks)
    scores['Hybrid']['prec'].append(p)
    scores['Hybrid']['ndcg'].append(n)

print(f"\n{gecerli_test_sayisi} adet uygun listeyle kapalı dünya testi tamamlandı.\n")

# 4. Sonuçları Ekrana Bas (Sadeleştirildi)
print("--- TEST SONUÇLARI (POSTER İÇİN) ---")
for model_name, metrics in scores.items():
    if len(metrics['prec']) > 0:
        avg_prec = np.mean(metrics['prec']) * 100
        avg_ndcg = np.mean(metrics['ndcg']) * 100
        avg_time = np.mean(metrics['times'])
        print(f"{model_name}:")
        print(f"  Doğruluk (nDCG): %{avg_ndcg:.2f}")
        print(f"  Hassasiyet (Precision): %{avg_prec:.2f}")
        print(f"  Hız (Latency): ~{avg_time:.0f} ms\n")