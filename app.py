
from collections import Counter
import os
import json
import logging
import math
import re
import time
import pickle
import uuid  # CRUCIAL: Para gerar IDs únicos de sessão
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from PyPDF2 import PdfReader

# ============================================================================
# 1. CONFIGURAÇÃO & HIPERPARÂMETROS
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CAMINHOS DE PERSISTÊNCIA ---
MEMORY_DIR = "primo_memory"
DB_FILE = os.path.join(MEMORY_DIR, "knowledge_base.parquet")
INDEX_FILE = os.path.join(MEMORY_DIR, "bm25_index.pkl")
HISTORY_FILE = os.path.join(MEMORY_DIR, "chat_history.json") # Novo arquivo de histórico

# Cria diretório de memória se não existir
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

# --- BRANDING ---
LOGO_PATH = "Primo_LOGO-removebg-preview.png" 
LOGO_PATH2 = "Logo_primo.png"

# --- TUNING DE SEGURANÇA & RETRIEVAL ---
MAX_SAFE_TOKENS = 80000     
MAX_SAFE_CHARS = MAX_SAFE_TOKENS * 4 
MAX_RETRIEVED_DOCS = 5      
CONTEXT_WINDOW_SIZE = 2      
MIN_CHUNK_LENGTH = 30        

LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.3            
BASE_URL = "https://api.deepseek.com"

# ============================================================================
# 2. ALGORITMO BM25 OTIMIZADO (SERIALIZÁVEL)
# ============================================================================

class SimpleBM25:
    """Implementação leve do BM25. Otimizada para ser 'picklable'."""
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.k1 = 1.5
        self.b = 0.75
        self.stopwords = {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 
            'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 
            'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 
            'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 
            'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 
            'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 
            'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 
            'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 
            'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 
            'estávamos', 'estavam', 'estivera', 'estivéramos', 'haja', 'hajamos', 'hajam', 'houve', 
            'houvemos', 'houveram', 'houvera', 'houvéramos', 'seja', 'sejamos', 'sejam', 'fosse', 
            'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 
            'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 
            'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 
            'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 
            'teria', 'teríamos', 'teriam'
        }
        self._initialize(corpus)

    def _initialize(self, corpus):
        total_length = 0
        for document in corpus:
            tokens = self._tokenize(document)
            self.doc_len.append(len(tokens))
            total_length += len(tokens)
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                self.idf[token] = self.idf.get(token, 0) + 1
        
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 1
        for token, freq in self.idf.items():
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def _tokenize(self, text: str) -> List[str]:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text) 
        tokens = text.split()
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            for token in query_tokens:
                if token not in freqs: continue
                freq = freqs[token]
                numerator = self.idf.get(token, 0) * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += numerator / denominator
        return scores

# --- SETUP SUPABASE ---
# Tenta pegar dos secrets do Streamlit (nuvem) ou variáveis de ambiente (local)
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

# Conecta ao banco (Singleton)
@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# NOVA GESTÃO DE CHAT (VIA SUPABASE)
# ============================================================================

def load_chat_history() -> Dict:
    """Baixa o histórico do Supabase."""
    supabase = get_supabase_client()
    if not supabase: return {}
    
    try:
        # Pega todas as sessões ordenadas pela última atualização
        response = supabase.table("chat_sessions").select("*").order("last_updated", desc=True).execute()
        
        # Converte a resposta do banco para o formato do nosso app (Dicionário)
        history = {}
        for row in response.data:
            history[row['session_id']] = {
                "title": row['title'],
                "timestamp": row['last_updated'], # ou created_at
                "messages": row['messages']
            }
        return history
    except Exception as e:
        st.error(f"Erro ao conectar na memória nuvem: {e}")
        return {}

def save_chat_session_remote(session_id: str, session_data: Dict):
    """Salva/Atualiza UMA sessão específica no Supabase."""
    supabase = get_supabase_client()
    if not supabase: return

    try:
        # Prepara o payload
        data = {
            "session_id": session_id,
            "title": session_data["title"],
            "messages": session_data["messages"],
            "last_updated": datetime.now().isoformat()
        }
        
        # Upsert: Se existe atualiza, se não cria
        supabase.table("chat_sessions").upsert(data).execute()
    except Exception as e:
        st.error(f"Erro ao salvar pensamento: {e}")

def create_new_chat_session():
    """Cria nova sessão local e remota."""
    new_id = str(uuid.uuid4())
    new_data = {
        "title": "Nova Conversa",
        "timestamp": datetime.now().isoformat(),
        "messages": []
    }
    
    # Atualiza estado local
    st.session_state.all_chats[new_id] = new_data
    st.session_state.active_session_id = new_id
    
    # Sincroniza nuvem
    save_chat_session_remote(new_id, new_data)

def delete_chat_session(session_id):
    """Remove do banco e do estado local."""
    supabase = get_supabase_client()
    
    # Remove local
    if session_id in st.session_state.all_chats:
        del st.session_state.all_chats[session_id]
    
    # Remove remoto
    if supabase:
        try:
            supabase.table("chat_sessions").delete().eq("session_id", session_id).execute()
        except Exception as e:
            st.error(f"Erro ao deletar da nuvem: {e}")

    # Lógica de redirecionamento
    if st.session_state.active_session_id == session_id:
        remaining_ids = list(st.session_state.all_chats.keys())
        if remaining_ids:
            st.session_state.active_session_id = remaining_ids[0]
        else:
            create_new_chat_session()
    st.rerun()

@st.cache_resource
def get_llm_client():
    return OpenAI(base_url=BASE_URL, api_key=os.getenv("DEEPSEEK_API_KEY"))

# ============================================================================
# 4. BUSCA & GERAÇÃO
# ============================================================================

def retrieve_context(query: str, df: pd.DataFrame, bm25: SimpleBM25) -> str:
    if df is None or bm25 is None: return ""
    
    if len(df) != bm25.corpus_size:
        bm25 = SimpleBM25(df['clean_text'].tolist())

    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:MAX_RETRIEVED_DOCS]
    top_indices = [i for i in top_indices if scores[i] > 1.0]

    if not top_indices: return ""

    expanded_indices = set()
    for idx in top_indices:
        start = max(0, idx - CONTEXT_WINDOW_SIZE)
        end = min(len(df), idx + CONTEXT_WINDOW_SIZE + 1)
        original_source = df.iloc[idx]['source_title']
        for i in range(start, end):
            if df.iloc[i]['source_title'] == original_source:
                expanded_indices.add(i)

    final_indices = sorted(list(expanded_indices))
    context_blocks = []
    current_chars = 0
    
    for idx in final_indices:
        row = df.iloc[idx]
        block = f"\n📺 FONTE: {row['source_title']} ({row['source_url']})\n- {row['clean_text']}\n"
        if current_chars + len(block) > MAX_SAFE_CHARS:
            context_blocks.append("\n⚠️ [SISTEMA: CONTEXTO LIMITE ATINGIDO] ⚠️")
            break
        context_blocks.append(block)
        current_chars += len(block)

    return "".join(context_blocks)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def generate_response(query: str, context: str):
    system_persona = """
     Você é o Gêmeo Digital do Thiago Nigro.
        DIRETRIZES:
        1. Seja detalhista e use o CONTEXTO fornecido.
        2. Cite os vídeos/fontes do contexto.
        3. Use a personalidade do Thiago (longo prazo)
        4. Seja extremamente detalhista, profundo e abrangente no máximo que você puder.
        5. Use apenas o CONTEXTO mais recente fornecido (Não use o conhecimento geral de treinamento do modelo) para responder.
        6. Sempre referencie nas suas respostas, o video mais recente utilizado para a mesma.
        7. Se o contexto for cortado, use o que tem disponível.
        8. Seja visionário, prático e conselheiro ou coach financeiro.
        9. Incorpore a essência intrínseca da alma do Thiago Nigro, use seu jeito de falar, suas gírias e sua personalidade única. Copie-o, Imite-o.
    """
    
    full_prompt = f"CONTEXTO RECUPERADO:\n{context}\n\nPERGUNTA DO PRIMO:\n{query}"
    client = get_llm_client()
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_persona}, {"role": "user", "content": full_prompt}],
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=8000
    )
    return stream

# ============================================================================
# 5. UI (STREAMLIT)
# ============================================================================

def main():
    st.set_page_config(page_title="Primo.AI | Gêmeo Digital", page_icon=LOGO_PATH2, layout="wide")
    
    # CSS Customizado para os "Cards" do Sidemenu
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: #f0f2f6; } 
            .stChatMessage { background-color: #1f2937; border: 1px solid #374151; }
            /* Estilo dos Botões de Histórico */
            div[data-testid="stSidebar"] button {
                text-align: left;
                width: 100%;
                border-radius: 8px;
                margin-bottom: 5px;
                padding: 10px;
                transition: all 0.2s;
            }
            div[data-testid="stSidebar"] button:hover {
                border-color: #fca311;
                color: #fca311;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- INICIALIZAÇÃO DE ESTADOS ---
    if "db" not in st.session_state:
        # Carrega Base de Conhecimento
        df_disk, bm25_disk = load_memory_from_disk()
        if df_disk is not None:
            st.session_state.db = df_disk
            st.session_state.bm25 = bm25_disk
        else:
            st.session_state.db = None
            st.session_state.bm25 = None
    
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = load_chat_history()
    
    if "active_session_id" not in st.session_state:
        # Se não tiver sessão ativa, cria uma ou pega a mais recente
        if st.session_state.all_chats:
            # Pega a última modificada (ou criada)
            st.session_state.active_session_id = list(st.session_state.all_chats.keys())[-1]
        else:
            create_new_chat_session()

    # --- SIDEBAR (HISTÓRICO & UPLOAD) ---
    with st.sidebar:
        c1, c2 = st.columns([1, 4])
        with c1:
            try: st.image(LOGO_PATH, width=50)
            except: st.write("🤖")
        with c2:
            st.title("Primo.AI")
            st.caption("Gêmeo Digital do Thiago Nigro | Desenvolvido por Gabriel Estrela")
            
        
        # Botão NOVA CONVERSA
        if st.button("➕ Nova Conversa", type="primary", use_container_width=True):
            create_new_chat_session()
            st.rerun()
            
        st.markdown("---")
        st.caption("HISTÓRICO DE CONVERSAS")

        # Renderiza lista de conversas (Cards)
        # Convertemos para lista para ordenar por timestamp (decrescente)
        chat_list = []
        for cid, cdata in st.session_state.all_chats.items():
            chat_list.append({"id": cid, "title": cdata["title"], "ts": cdata["timestamp"]})
        
        # Ordena: Mais recentes primeiro
        chat_list.sort(key=lambda x: x["ts"], reverse=True)

        for chat in chat_list:
            # Ícone diferente se for a conversa ativa
            icon = "🟢" if chat["id"] == st.session_state.active_session_id else "💬"
            label = f"{icon} {chat['title']}"
            
            # Usamos o ID como key do botão para unicidade
            if st.button(label, key=f"btn_{chat['id']}"):
                st.session_state.active_session_id = chat["id"]
                st.rerun()

        st.markdown("---")
        with st.expander("🛠️ Status do Sistema"):
            st.write(f"Conhecimento: {len(st.session_state.db) if st.session_state.db is not None else 0} chunks")
            st.write(f"Cloud: Supabase Online ✅")
            if st.button("🗑️ Deletar Conversa Atual"):
                delete_chat_session(st.session_state.active_session_id)

    # --- MAIN CHAT AREA ---
    current_id = st.session_state.active_session_id
    current_chat_data = st.session_state.all_chats[current_id]
    messages = current_chat_data["messages"]

    # Renderiza o histórico de mensagens da sessão atual
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Logica de Entrada de Usuário
    if prompt := st.chat_input("Pergunte ao Primo sobre investimentos, foco ou gestão..."):
        
        # 1. Update UI & Local State
        new_user_message = {"role": "user", "content": prompt}
        messages.append(new_user_message)
        st.session_state.all_chats[current_id]["messages"] = messages
        
        # 2. Auto-titulação (Somente se for a primeira interação da conversa)
        if len(messages) == 1:
            title = prompt[:30] + ("..." if len(prompt) > 30 else "")
            st.session_state.all_chats[current_id]["title"] = title
            
        # 3. Sincronização Remota (Início)
        st.session_state.all_chats[current_id]["timestamp"] = datetime.now().isoformat()
        save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Geração de Resposta com RAG
        with st.chat_message("assistant"):
            resp_container = st.empty()
            
            with st.spinner("🔍 Consultando fontes e Skin in the Game..."):
                context = retrieve_context(prompt, st.session_state.db, st.session_state.bm25)
            
            if not context:
                msg_fail = "Primo, não encontrei referências específicas disso nas minhas mentorias gravadas. Mas como Thiago Nigro, eu diria para focar no longo prazo!"
                resp_container.markdown(msg_fail)
                messages.append({"role": "assistant", "content": msg_fail})
                st.session_state.all_chats[current_id]["messages"] = messages
                save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
            else:
                full_res = ""
                try:
                    stream = generate_response(prompt, context)
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_res += content
                        resp_container.markdown(full_res + "▌")
                    
                    resp_container.markdown(full_res)
                    
                    # 5. Finalização & Sincronização Remota (Fim)
                    messages.append({"role": "assistant", "content": full_res})
                    st.session_state.all_chats[current_id]["messages"] = messages
                    save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
                    
                    # Rerun para atualizar o título no sidebar se foi a primeira vez
                    if len(messages) <= 2:
                        st.rerun()

                except Exception as e:
                    st.error(f"Erro na conexão com DeepSeek: {e}")

if __name__ == "__main__":
    main()