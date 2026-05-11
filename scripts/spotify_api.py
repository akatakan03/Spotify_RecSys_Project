import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyAPIHelper:
    def __init__(self, client_id, client_secret):
        # Güvenlik kimliklerinizle Spotify'a giriş yapıyoruz (Authentication)
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def get_track_details(self, track_uri):
        """ Şarkının URI kodunu alıp albüm kapağını ve bağlantı linkini getirir """
        try:
            # URI temizleme işlemi (Clean URI)
            if not track_uri.startswith('spotify:track:'):
                clean_uri = track_uri.split(':')[-1] if ':' in track_uri else track_uri
                track_uri = f'spotify:track:{clean_uri}'
                
            track_info = self.sp.track(track_uri)
            
            # get() metodu kullanarak Dictionary (Sözlük) içinde güvenli arama yapıyoruz
            return {
                'cover_url': track_info['album']['images'][0]['url'] if track_info.get('album', {}).get('images') else None,
                'preview_url': track_info.get('preview_url'), 
                'spotify_url': track_info.get('external_urls', {}).get('spotify')
            }
        except Exception as e:
            print(f"Spotify API Hatası ({track_uri}): {e}")
            return None