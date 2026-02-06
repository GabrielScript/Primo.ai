import os
import json
import time
import requests
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
load_dotenv()

# --- CONFIGURAÇÃO ---
load_dotenv()

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")
CHANNEL_HANDLE = "primorico"  # Pode ser alterado conforme necessidade
MAX_VIDEOS = 50  # Número máximo de vídeos a processar
SORT_BY = "latest"  # Opções: 'latest', 'popular'

# --- CLIENTE HTTP ROBUSTO ---

def create_session_with_retry() -> requests.Session:
    """Cria uma sessão HTTP com política de retries automática."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# --- FUNÇÕES DE EXTRAÇÃO ---

def fetch_youtube_video_list(session: requests.Session, handle: str, limit: int = 10, sort: str = 'latest') -> List[Dict]:
    """
    Busca a lista de vídeos de um canal.
    """
    logging.info(f"📡 Buscando vídeos do canal @{handle} ({sort})...")
    
    # URL hipotética baseada em APIs de scraper comuns. 
    url = "https://api.scrapecreators.com/v1/youtube/channel/videos"
    
    params = {
        "handle": handle,
        "limit": limit,
        "sort": sort,
        "api_key": SCRAPECREATORS_API_KEY
    }
    
    try:
        # Simulação de chamada.
        response = session.get(url, params=params, timeout=30)
        # Se 404/403, retorna lista vazia para não quebrar o app
        if response.status_code >= 400:
             logging.error(f"Erro API ScrapeCreators: {response.status_code} - {response.text}")
             return []
             
        data = response.json()
        videos = data.get('videos', data.get('data', []))
        logging.info(f"✅ Encontrados {len(videos)} vídeos.")
        return videos
    except Exception as e:
        logging.error(f"❌ Erro ao buscar lista de vídeos: {e}")
        return []

def fetch_video_transcript(session: requests.Session, video_id: str) -> Optional[str]:
    """Busca a transcrição completa de um vídeo específico."""
    url = f"https://api.scrapecreators.com/v1/youtube/video/transcript"
    params = {
        "video_id": video_id,
        "api_key": SCRAPECREATORS_API_KEY
    }
    
    try:
        response = session.get(url, params=params, timeout=20)
        if response.status_code == 404:
            return None
            
        data = response.json()
        transcript = data.get('transcript', "")
        
        if isinstance(transcript, list):
            full_text = " ".join([seg.get('text', '') for seg in transcript])
            return full_text
            
        return transcript
    except Exception as e:
        logging.warning(f"⚠️ Falha na transcrição do vídeo {video_id}: {e}")
        return None

# --- EXPORTABLE FUNCTION (SYNC) ---

def update_channel_data(existing_ids: List[str] = None) -> List[Dict]:
    """
    Função modular para buscar NOVOS vídeos.
    Retorna apenas os vídeos que foram processados com sucesso.
    """
    if not SCRAPECREATORS_API_KEY:
        logging.warning("⚠️ SCRAPECREATORS_API_KEY não configurada.")
        return []

    if existing_ids is None:
        existing_ids = []

    session = create_session_with_retry()
    
    # 1. Busca lista (pode buscar mais para garantir se houve muito upload recente)
    videos = fetch_youtube_video_list(session, CHANNEL_HANDLE, limit=20, sort="latest")
    
    new_data = []
    
    for video in videos:
        video_id = video.get('id') or video.get('videoId')
        if not video_id: continue
        
        # Incremental Check: Pula se já existe
        if video_id in existing_ids:
            continue
            
        logging.info(f"🆕 Processando NOVO vídeo: {video.get('title')}")
        transcript = fetch_video_transcript(session, video_id)
        
        if transcript:
            entry = {
                "source_id": video_id,
                "source_title": video.get('title', 'Novo Vídeo'),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "published_at": video.get('publishedAt', ''),
                "text": transcript
            }
            new_data.append(entry)
            # Respeita rate limit simples
            time.sleep(1)
            
    return new_data

# --- FLUXO PRINCIPAL (CLI) ---

def main():
    new_videos = update_channel_data()
    
    if new_videos:
        filename = f"transcricoes_{CHANNEL_HANDLE}_{len(new_videos)}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(new_videos, f, ensure_ascii=False, indent=4)
        print(f"Salvou {len(new_videos)} novos vídeos.")
    else:
        print("Nenhum vídeo novo processado.")

if __name__ == "__main__":
    main()
