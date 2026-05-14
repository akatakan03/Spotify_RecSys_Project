import polars as pl
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

class SpotifyRecommenderPipeline:
    def __init__(self):
        print("🚀 Ana Boru Hattı (Pipeline) Başlatılıyor. Büyük Veriler Belleğe Yükleniyor...")
        
        # 1. Dosya Yolları
        matrix_path = 'data/processed/sparse_cooccurrence.npz'
        mapping_path = 'data/processed/uri_mapping.pkl'
        metadata_path = 'data/processed/offline_track_metadata.parquet'
        features_path = 'data/processed/audio_features_v2.parquet'

        # 2. Matris ve Haritaları Yükleme
        self.sparse_matrix = load_npz(matrix_path)
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
            self.uri_to_idx = mapping['uri_to_idx']
            self.idx_to_uri = mapping['idx_to_uri']

        # 3. Veri Çerçevelerini (Dataframes) Hazırlama
        self.df_metadata = pl.read_parquet(metadata_path).to_pandas()
        df_features = pl.read_parquet(features_path).to_pandas()
        self.df_merged = pd.merge(df_features, self.df_metadata, on='track_uri')

        # 4. Veri Temizliği ve Standardization (Ölçeklendirme)
        self.feature_cols = ['bpm', 'energy', 'spectral_centroid', 'zero_crossing_rate', 'spectral_rolloff', 'harmonic_ratio', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5']
        
        # ---> İŞTE HAYAT KURTARAN SATIR: Eksik/Bozuk verileri (NaN) sıfır ile dolduruyoruz (Data Cleaning) <---
        self.df_merged[self.feature_cols] = self.df_merged[self.feature_cols].fillna(0)

        self.scaler = StandardScaler()
        self.df_merged[self.feature_cols] = self.scaler.fit_transform(self.df_merged[self.feature_cols])

        # 5. Senaryo (NLP) İçin Semantic Features Hazırlığı
        self.df_metadata['text_tags'] = (self.df_metadata['track_name'] + " " + 
                                         self.df_metadata['artist_name'] + " " + 
                                         self.df_metadata['album_name']).str.lower()
        
        # Metinlerde de olası NaN değerlerini temizliyoruz
        self.df_metadata['text_tags'] = self.df_metadata['text_tags'].fillna("")

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_metadata['text_tags'])
        
        print("✅ Sistem Hazır! Tüm modüller başarıyla entegre edildi.\n")

    def recommend_hybrid(self, seed_uris, w_audio=0.5, w_co=0.5, top_n=10):
        """ Şarkılara Göre Hibrit Öneri (Audio + Collaborative) """
        seed_indices = [self.uri_to_idx[uri] for uri in seed_uris if uri in self.uri_to_idx]
        if not seed_indices:
            return []

        # 1. Content-Based (Audio) Puanı
        user_profile = self.df_merged[self.df_merged['track_uri'].isin(seed_uris)][self.feature_cols].mean().values.reshape(1, -1)
        audio_sims = (cosine_similarity(user_profile, self.df_merged[self.feature_cols].values)[0] + 1) / 2
        
        # 2. Collaborative (Co-occurrence) Puanı (TÜM MATRİS)
        co_scores_full = np.array(self.sparse_matrix[seed_indices, :].sum(axis=0)).flatten()
        
        # 3. BOYUT HİZALAMA (Dimensionality Alignment)
        merged_uris = self.df_merged['track_uri'].tolist()
        merged_matrix_indices = [self.uri_to_idx[uri] for uri in merged_uris]
        co_scores = co_scores_full[merged_matrix_indices]
        
        if co_scores.max() > 0:
            co_scores = co_scores / co_scores.max()
        
        # 4. Hybrid Equation
        final_scores = (audio_sims * w_audio) + (co_scores * w_co)
        results_idx = np.argsort(final_scores)[::-1]
        
        oneriler = []
        for idx in results_idx:
            uri = self.df_merged.iloc[idx]['track_uri'] 
            if uri not in seed_uris:
                sarki = self.df_metadata[self.df_metadata['track_uri'] == uri].iloc[0]
                oneriler.append({
                    'uri': uri,
                    'name': sarki['track_name'],
                    'artist': sarki['artist_name'],
                    'score': final_scores[idx],
                    'type': 'hybrid'
                })
            if len(oneriler) == top_n:
                break
        return oneriler

    def recommend_scenario(self, query_text, top_n=10):
        """ Metne Göre Öneri (Scenario-Aware NLP) """
        query_vec = self.vectorizer.transform([query_text.lower()])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        results_idx = np.argsort(sim_scores)[::-1][:top_n]
        
        if sim_scores[results_idx[0]] == 0:
            return [] 
            
        oneriler = []
        for idx in results_idx:
            sarki = self.df_metadata.iloc[idx]
            oneriler.append({
                'uri': sarki['track_uri'],
                'name': sarki['track_name'],
                'artist': sarki['artist_name'],
                'score': sim_scores[idx],
                'type': 'scenario'
            })
        return oneriler

if __name__ == "__main__":
    pipeline = SpotifyRecommenderPipeline()
    print("Test başarılı!")