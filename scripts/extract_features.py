import os
import librosa
import numpy as np
import polars as pl
import warnings

# Librosa'nın gereksiz uyarılarını gizleyelim ki terminalimiz temiz kalsın
warnings.filterwarnings('ignore')

audio_dir = 'data/audio_snippets'
output_file = 'data/processed/audio_features_v2.parquet' # Dosya adını güncelledik

print("Gelişmiş Ses Analiz Motoru (V2) Başlatılıyor...")
results = []

for filename in os.listdir(audio_dir):
    if filename.endswith(".mp3"):
        filepath = os.path.join(audio_dir, filename)
        track_id = filename.replace('.mp3', '')
        track_uri = f"spotify:track:{track_id}"
        
        print(f"Röntgen çekiliyor: {track_id}...")
        
        try:
            # Sesi yüklüyoruz (Yine 30 saniye)
            y, sr = librosa.load(filepath, sr=22050, duration=30)
            
            # --- 1. TEMEL ÖZELLİKLER ---
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
            
            rms = librosa.feature.rms(y=y)
            energy = float(np.mean(rms))
            
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid = float(np.mean(cent))
            
            # --- 2. YENİ EKLENEN AKADEMİK ÖZELLİKLER ---
            # Gürültü/Sertlik Oranı (ZCR)
            zcr = librosa.feature.zero_crossing_rate(y=y)
            zero_crossing_rate = float(np.mean(zcr))
            
            # Frekans Sınırı (Rolloff)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff = float(np.mean(rolloff))
            
            # Armonik ve Perküsif (Vurmalı) Ayrımı
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_mean = float(np.mean(librosa.feature.rms(y=y_harmonic)))
            percussive_mean = float(np.mean(librosa.feature.rms(y=y_percussive)))
            
            # Sesin DNA'sı (MFCC - İlk 5 katsayı)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
            mfcc_means = np.mean(mfccs, axis=1) # 5 farklı boyut verecek
            
            # Tüm özellikleri tek bir sözlükte (Dictionary) topluyoruz
            feature_dict = {
                'track_uri': track_uri,
                'bpm': bpm,
                'energy': energy,
                'spectral_centroid': spectral_centroid,
                'zero_crossing_rate': zero_crossing_rate,
                'spectral_rolloff': spectral_rolloff,
                'harmonic_ratio': harmonic_mean / (percussive_mean + 0.001), # Sıfıra bölünme hatasını önlemek için 0.001
                'mfcc_1': float(mfcc_means[0]),
                'mfcc_2': float(mfcc_means[1]),
                'mfcc_3': float(mfcc_means[2]),
                'mfcc_4': float(mfcc_means[3]),
                'mfcc_5': float(mfcc_means[4]),
            }
            
            results.append(feature_dict)
            print(f" -> Başarılı! (Toplam 11 Özellik Çıkarıldı)")
            
        except Exception as e:
            print(f" -> [Hata] Analiz edilemedi: {e}")

if results:
    df_features = pl.DataFrame(results)
    df_features.write_parquet(output_file)
    print(f"\nMuazzam! {len(df_features)} şarkının 11 Boyutlu (11D) ses haritası çıkarıldı.")
    print(f"Yeni Dosya: {output_file}")
else:
    print("Hiçbir özellik çıkarılamadı.")