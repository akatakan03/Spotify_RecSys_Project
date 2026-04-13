import os
import time
import polars as pl
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# 1. YENİ ALDIĞIN ANAHTARLARI BURAYA GİR
CLIENT_ID = '7e58c74eb38f481c918fcda316eed479'
CLIENT_SECRET = 'c4c97cb892de4d48bd738e151dd88f67'
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

master_file = 'data/subset_100k_tracks.parquet'
output_file = 'data/processed/track_metadata.parquet'

df_subset = pl.read_parquet(master_file)
all_uris = df_subset['track_uri'].to_list()
valid_uris = [uri for uri in all_uris if str(uri).startswith('spotify:track:')]

if os.path.exists(output_file):
    df_saved = pl.read_parquet(output_file)
    saved_uris = set(df_saved['track_uri'].to_list())
    remaining_uris = [uri for uri in valid_uris if uri not in saved_uris]
    print(f"Kaldığı yerden devam ediliyor. Mevcut: {len(saved_uris)}, Kalan: {len(remaining_uris)}")
else:
    remaining_uris = valid_uris
    print(f"Sıfırdan başlanıyor. Temizlenmiş Toplam: {len(remaining_uris)} şarkı.")

if not remaining_uris:
    print("Tüm şarkılar zaten indirilmiş!")
    exit()

batch_size = 50
chunks = [remaining_uris[i:i + batch_size] for i in range(0, len(remaining_uris), batch_size)]

print(f"Veri çekimi başlıyor ({len(chunks)} paket)...")

for i, chunk in enumerate(chunks):
    batch_results = []
    
    try:
        results = sp.tracks(chunk, market='TR')
        for track in results['tracks']:
            if track is not None:
                batch_results.append({
                    'track_uri': track['uri'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                })
        
        # Başarılı olursa veriyi diske yaz
        if batch_results:
            df_new = pl.DataFrame(batch_results)
            if os.path.exists(output_file):
                df_final = pl.concat([pl.read_parquet(output_file), df_new])
                df_final.write_parquet(output_file)
            else:
                df_new.write_parquet(output_file)
                
        # Güvenli bekleme süresi
        time.sleep(0.5)
        
        if (i + 1) % 10 == 0:
            print(f"İlerleme: {i+1}/{len(chunks)} paket başarıyla tamamlandı.")
            
    except SpotifyException:
        # İŞTE YENİ STRATEJİMİZ: ZEHİRLİ PAKETİ DİREKT ÇÖPE AT!
        print(f"Paket {i+1} zehirli! API limitini korumak için tüm paket çöpe atıldı, sonrakine geçiliyor...")
        time.sleep(0.5) # Banlanmamak için yine de yarım saniye bekle
        continue 
        
    except Exception as e:
        print(f"Genel hata oluştu, 5 saniye bekleniyor: {e}")
        time.sleep(5)
        continue

print("Tüm metadata başarıyla toplandı (Zehirli paketler elendi)!")