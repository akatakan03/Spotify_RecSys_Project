import os
import time
import polars as pl
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# 1. Yeni anahtarlarını buraya yapıştır (Tırnakları unutma!)
CLIENT_ID = '7e58c74eb38f481c918fcda316eed479'
CLIENT_SECRET = 'c4c97cb892de4d48bd738e151dd88f67'
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# 2. Dosya Yolları
master_file = 'data/subset_100k_tracks.parquet'
output_file = 'data/processed/track_metadata.parquet'

# 3. Veriyi Oku ve Filtrele
df_subset = pl.read_parquet(master_file)
all_uris = df_subset['track_uri'].to_list()
valid_uris = [uri for uri in all_uris if str(uri).startswith('spotify:track:')]

# 4. Checkpoint Sistemi
if os.path.exists(output_file):
    df_saved = pl.read_parquet(output_file)
    saved_uris = set(df_saved['track_uri'].to_list())
    remaining_uris = [uri for uri in valid_uris if uri not in saved_uris]
    print(f"Kaldığı yerden devam... Mevcut: {len(saved_uris)}, Kalan: {len(remaining_uris)}")
else:
    remaining_uris = valid_uris
    print(f"Sıfırdan başlanıyor... Toplam: {len(remaining_uris)}")

if not remaining_uris:
    print("İşlem zaten tamamlanmış!")
    exit()

# 5. Küçük Paketler (Veri kaybını önlemek için 20 idealdir)
batch_size = 20
chunks = [remaining_uris[i:i + batch_size] for i in range(0, len(remaining_uris), batch_size)]

print(f"Veri çekimi başlıyor ({len(chunks)} paket)...")

for i, chunk in enumerate(chunks):
    batch_results = []
    
    try:
        # DİKKAT: market parametresi tamamen kaldırıldı!
        results = sp.tracks(chunk)
        
        for track in results['tracks']:
            if track: # Şarkı silinmişse bile bu blok sayesinde hata almayız
                batch_results.append({
                    'track_uri': track['uri'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                })
        
        if batch_results:
            df_new = pl.DataFrame(batch_results)
            if os.path.exists(output_file):
                df_final = pl.concat([pl.read_parquet(output_file), df_new])
                df_final.write_parquet(output_file)
            else:
                df_new.write_parquet(output_file)
        
        time.sleep(0.7) # Spotify'ı kızdırmayalım
        
    except SpotifyException as e:
        if e.http_status == 403:
            print(f"Paket {i+1} lisans engeline takıldı, bu paket atlanıyor.")
        elif e.http_status == 429:
            print("Rate Limit! 30 saniye mola...")
            time.sleep(30)
        continue
    except Exception as e:
        print(f"Bilinmeyen hata: {e}")
        time.sleep(5)
        continue

    if (i + 1) % 10 == 0:
        print(f"İlerleme: {i+1}/{len(chunks)} paket bitti.")

print("İşlem başarıyla tamamlandı!")