import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

print("NLP Tabanlı Senaryo Farkındalığı (Scenario-Awareness) Motoru Başlatılıyor...\n")

# 1. Veriyi Yükleme
df_metadata = pl.read_parquet('data/processed/offline_track_metadata.parquet').to_pandas()

# 2. Semantic Features (Anlamsal Özellikler) Oluşturma
df_metadata['text_tags'] = df_metadata['track_name'] + " " + df_metadata['artist_name'] + " " + df_metadata['album_name']

df_metadata['text_tags'] = df_metadata['text_tags'].str.lower()

# 3. TF-IDF Vectorizer (Kelimeleri Matematiğe Çevirme)
print("Müzik kütüphanesindeki tüm kelimeler öğreniliyor (TF-IDF)...")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_metadata['text_tags'])

# 4. Senaryo Tabanlı Öneri Fonksiyonu
def senaryoya_gore_oner(senaryo_metni, onerilecek_sayi=10):
    print("-" * 50)
    print(f"🔎 Kullanıcı Senaryosu (Query): '{senaryo_metni}'")
    print("-" * 50)
    
    senaryo_vektoru = vectorizer.transform([senaryo_metni.lower()])
    
    benzerlik_skorlari = cosine_similarity(senaryo_vektoru, tfidf_matrix)[0]
    
    skorlar = list(enumerate(benzerlik_skorlari))
    sirali_skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    en_yakinlar = sirali_skorlar[:onerilecek_sayi]
    
    if sirali_skorlar[0][1] == 0:
        print("Bu senaryoya uygun hiçbir anlamsal eşleşme bulunamadı.")
        return
        
    print("💡 Senaryo İçin En İyi Eşleşmeler (Semantic Search):")
    for i, (idx, skor) in enumerate(en_yakinlar):
        isim = df_metadata.iloc[idx]['track_name']
        sanatci = df_metadata.iloc[idx]['artist_name']
        album = df_metadata.iloc[idx]['album_name']
        print(f"{i+1}. {isim} - {sanatci} (Skor: %{skor*100:.1f}) | Albüm: {album}")

# TEST AŞAMASI
test_senaryosu = "acoustic guitar"
senaryoya_gore_oner(test_senaryosu)