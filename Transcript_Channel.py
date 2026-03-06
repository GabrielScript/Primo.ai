import requests
import json
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
# Tente carregar o python-dotenv, mas não falhe se não estiver instalado
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Nota: 'python-dotenv' não encontrado. As variáveis de ambiente devem ser definidas manualmente.")

# ==========================
# CONFIGURAÇÕES DA MISSÃO
# ==========================
# ⭐ Defina sua chave de API aqui ou via variável de ambiente "SCRAPECREATORS_API_KEY"
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY")
# ⭐ Defina o "handle" (nome) do canal
CHANNEL_HANDLE = "primorico" # Canal do Primo Rico
# ⭐ Defina quantos vídeos você deseja (Meta de Custo: ~MAX_VIDEOS + (MAX_VIDEOS / 30))
MAX_VIDEOS = 500
# ⭐ Ordenar por 'latest' (mais recente) ou 'popular'
SORT_BY = "latest"

# ==========================
# SESSÃO DE REQUISIÇÕES ROBUSTA
# ==========================
def create_session_with_retry():
    """
    Cria uma sessão HTTP robusta com política de retry exponencial.
    Isso é crucial para longas operações de scraping.
    """
    session = requests.Session()
    # Define a política de retry
    retry = Retry(
        total=5,                # Número total de tentativas
        read=5,                 # Tentar novamente em erros de leitura
        connect=5,              # Tentar novamente em erros de conexão
        backoff_factor=2,       # Fator exponencial (ex: 2s, 4s, 8s, 16s)
        status_forcelist=[429, 500, 502, 503, 504], # Status HTTP que disparam o retry
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# ==========================
# FASE 1: COLETAR IDs DOS VÍDEOS (O "1" de N+1)
# ==========================
def fetch_youtube_video_list(session, handle, api_key, max_videos):
    """
    Busca a lista de vídeos de um canal, página por página.
    Custo: ~1 crédito por página (aprox. 30 vídeos/página).
    """
    url = "https://api.scrapecreators.com/v1/youtube/channel-videos"
    headers = {"x-api-key": api_key}
    
    params = {
        "handle": handle,
        "sort": SORT_BY,
        "includeExtras": "true" # Como por sua nova documentação
    }
    
    all_videos = []
    continuation_token = None
    
    print(f"🎯 FASE 1: Coletando lista de {max_videos} vídeos do canal '{handle}'...")
    print(f"Custo estimado da Fase 1: ~{ (max_videos // 30) + 1 } créditos.")
    
    # tqdm é uma barra de progresso visual
    with tqdm(total=max_videos, desc="📺 Coletando IDs de vídeos") as pbar:
        while len(all_videos) < max_videos:
            if continuation_token:
                params["continuationToken"] = continuation_token
            
            try:
                # Faz a chamada de API com timeout de 60s
                response = session.get(url, headers=headers, params=params, timeout=60)
                # Lança um erro se o status for ruim (ex: 401, 403, 500)
                response.raise_for_status()
                
                data = response.json()
                videos_on_page = data.get("videos", [])
                
                if not videos_on_page:
                    print(f"\n⚠️ Nenhum vídeo novo encontrado nesta página. Fim do canal?")
                    break
                
                # Calcula quantos vídeos faltam para atingir a meta
                remaining_needed = max_videos - len(all_videos)
                # Adiciona apenas os vídeos necessários para não ultrapassar a meta
                videos_to_add = videos_on_page[:remaining_needed]
                all_videos.extend(videos_to_add)
                
                pbar.update(len(videos_to_add)) # Atualiza a barra de progresso
                
                # Se já atingimos a meta, paramos
                if len(all_videos) >= max_videos:
                    break
                
                # Pega o token para a *próxima* página
                continuation_token = data.get("continuationToken")
                if not continuation_token:
                    print(f"\n⚠️ Fim do canal atingido. Total de {len(all_videos)} vídeos encontrados.")
                    break
                
                # Pequena pausa para não sobrecarregar a API
                time.sleep(0.5) 
                    
            except requests.exceptions.HTTPError as http_err:
                print(f"\n❌ Erro HTTP na Fase 1: {http_err}")
                if http_err.response.status_code == 401:
                    print("ERRO: Chave de API inválida ou ausente (401 Unauthorized). Verifique sua 'x-api-key'.")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado na Fase 1: {e}")
                break
    
    print(f"✅ Fase 1 concluída. {len(all_videos)} IDs de vídeo coletados.")
    return all_videos

# ==========================
# FASE 2: BUSCAR TRANSCRIÇÕES (O "N" de N+1)
# ==========================
def fetch_video_transcript(session, video_item, api_key):
    """
    Busca a transcrição para um ÚNICO vídeo.
    Custo: 1 crédito por chamada.
    """
    # Usamos o endpoint otimizado para transcrições, como no seu script original
    url = "https://api.scrapecreators.com/v1/youtube/video/transcript"
    headers = {"x-api-key": api_key}
    
    # O endpoint de transcrição é mais eficiente se passarmos a URL
    video_url = video_item.get("url")
    if not video_url:
        return None # Pula se não tiver URL
        
    params = {"url": video_url}
    
    try:
        response = session.get(url, headers=headers, params=params, timeout=60)
        
        # O endpoint de transcrição pode retornar 200 mesmo sem transcrição
        if response.status_code == 200:
            data = response.json()
            
            # Verificamos se a API teve sucesso E se a transcrição existe
            if data.get("success") and data.get("transcript"):
                
                # Retornamos um objeto limpo com os dados da Fase 1 + Transcrição
                return {
                    "video_id": video_item.get("id"),
                    "title": video_item.get("title"),
                    "description": video_item.get("description"), # Vem da Fase 1 (includeExtras=true)
                    "url": video_url,
                    "views": video_item.get("viewCountInt"), # Vem da Fase 1
                    "likes": video_item.get("likeCount"), # Vem da Fase 1 (se disponível)
                    "published_at": video_item.get("publishedTime"), # Vem da Fase 1
                    "duration_sec": video_item.get("lengthSeconds"), # Vem da Fase 1
                    "transcript": data.get("transcript") # A "informação suculenta" da Fase 2
                }
        
        # Se o status não for 200 ou a transcrição falhar, retorna None
        return None
        
    except Exception as e:
        # Não paramos o script por um vídeo, apenas registramos o erro
        print(f"\n⚠️ Erro ao buscar transcrição para {video_url}: {e}")
        return None

# ==========================
# FUNÇÃO DE AUTO-SYNC (chamada pelo app.py)
# ==========================
def update_channel_data(existing_ids=None):
    """
    Busca APENAS vídeos novos que não estão na base.
    Retorna lista de dicts com video_id, source_title, url, transcript (texto).
    """
    if not SCRAPECREATORS_API_KEY:
        print("⚠️ SCRAPECREATORS_API_KEY não configurada. Auto-sync desabilitado.")
        return []
    
    if existing_ids is None:
        existing_ids = []
    
    existing_set = set(existing_ids)
    session = create_session_with_retry()
    
    # Busca os vídeos mais recentes (limite menor para sync rápido)
    video_list = fetch_youtube_video_list(session, CHANNEL_HANDLE, SCRAPECREATORS_API_KEY, max_videos=60)
    
    if not video_list:
        return []
    
    # Filtra apenas vídeos NOVOS
    new_videos = [v for v in video_list if v.get("id") not in existing_set]
    
    if not new_videos:
        print("✅ Nenhum vídeo novo encontrado.")
        return []
    
    print(f"🆕 {len(new_videos)} vídeos novos detectados. Buscando transcrições...")
    
    results = []
    for video_item in new_videos:
        transcript_data = fetch_video_transcript(session, video_item, SCRAPECREATORS_API_KEY)
        if transcript_data:
            # Formata para o padrão esperado pelo app.py (KnowledgeBase.process_and_index)
            transcript_text = transcript_data.get("transcript", "")
            if isinstance(transcript_text, list):
                transcript_text = " ".join([seg.get("text", "") for seg in transcript_text if isinstance(seg, dict)])
            
            results.append({
                "video_id": transcript_data.get("video_id", ""),
                "source_title": transcript_data.get("title", "Vídeo Sem Título"),
                "url": transcript_data.get("url", ""),
                "text": transcript_text
            })
        time.sleep(0.3)
    
    print(f"✅ Auto-sync: {len(results)} novos vídeos processados.")
    return results

# ==========================
# FUNÇÃO PRINCIPAL (MAIN)
# ==========================
def main():
    
    print("\n" + "="*80)
    print(f"🚀 INICIANDO COLETA DE TRANSCRIÇÕES".center(80))
    print(f"Canal: {CHANNEL_HANDLE} | Meta: {MAX_VIDEOS} vídeos".center(80))
    print("="*80 + "\n")
    
    if SCRAPECREATORS_API_KEY == "SUA_CHAVE_API_AQUI" or not SCRAPECREATORS_API_KEY:
        print("❌ ERRO CRÍTICO: 'SCRAPECREATORS_API_KEY' não está definida.")
        print("Por favor, defina a variável de ambiente ou edite o script.")
        return

    # Cria a sessão que será usada para todas as chamadas
    session = create_session_with_retry()
    
    start_time_total = time.time()
    
    # --- FASE 1 ---
    video_list = fetch_youtube_video_list(session, CHANNEL_HANDLE, SCRAPECREATORS_API_KEY, MAX_VIDEOS)
    
    if not video_list:
        print("❌ Nenhuma lista de vídeos foi coletada. Encerrando.")
        return
        
    # --- FASE 2 ---
    print(f"\n🎯 FASE 2: Buscando {len(video_list)} transcrições (1 por 1)...")
    print(f"Custo estimado da Fase 2: {len(video_list)} créditos.")
    
    videos_with_transcripts = []
    
    # Barra de progresso para a Fase 2
    with tqdm(total=len(video_list), desc="📝 Processando transcrições") as pbar:
        for video_item in video_list:
            transcript_data = fetch_video_transcript(session, video_item, SCRAPECREATORS_API_KEY)
            
            if transcript_data:
                videos_with_transcripts.append(transcript_data)
            
            # Pausa muito breve para ser um bom "cidadão da web"
            time.sleep(0.3)
            pbar.update(1) # Atualiza a barra de progresso
    
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    
    # --- FASE 3: SALVAR E RELATÓRIO ---
    print("\n" + "="*80)
    print("🎉 PROCESSO CONCLUÍDO!".center(80))
    print("="*80)
    
    taxa_sucesso = (len(videos_with_transcripts) / len(video_list) * 100) if video_list else 0
    custo_total_estimado = (len(video_list) // 30 + 1) + len(video_list)
    
    print(f"\n📊 --- Relatório da Missão --- 📊")
    print(f"  • Tempo Total:    {total_duration:.2f} segundos")
    print(f"  • Custo Total Est.: ~{custo_total_estimado} créditos")
    print(f"  • Vídeos Alvo:    {len(video_list)}")
    print(f"  • Transcrições Ok: {len(videos_with_transcripts)}")
    print(f"  • Taxa de Sucesso: {taxa_sucesso:.1f}%")
    
    # Salvar os resultados em um arquivo JSON
    if videos_with_transcripts:
        output_filename = f"transcricoes_{CHANNEL_HANDLE}_{MAX_VIDEOS}.json"
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(videos_with_transcripts, f, ensure_ascii=False, indent=2)
            print(f"\n💾 {len(videos_with_transcripts)} transcrições salvas em '{output_filename}'")
        except Exception as e:
            print(f"\n❌ Erro ao salvar arquivo JSON: {e}")

if __name__ == "__main__":
    main()