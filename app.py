# app.py
# ----------------------------------------------------------------------------
# ARQUITETURA: RAG Local (Git) + DeepSeek + Persistência de Chat (Supabase)
# AUTOR: Ph.D. Assistant & User
# VERSÃO: 9.0 (Gold Master - Cloud Ready)
# ----------------------------------------------------------------------------

import os
import json
import logging
import math
import re
import time
import pickle
import uuid
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from supabase import create_client, Client

# ============================================================================
# 1. CONFIGURAÇÃO & HIPERPARÂMETROS
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CAMINHOS DE PERSISTÊNCIA (MEMÓRIA ESTÁTICA) ---
# No Streamlit Cloud, isso virá do seu repositório Git
MEMORY_DIR = "primo_memory"
DB_FILE = os.path.join(MEMORY_DIR, "knowledge_base.parquet")
INDEX_FILE = os.path.join(MEMORY_DIR, "bm25_index.pkl")

# --- BRANDING ---
# Certifique-se que esses arquivos estão na raiz do repo
LOGO_PATH = "Primo_LOGO-removebg-preview.png" 
LOGO_PATH2 = "Logo_primo.png"

# --- TUNING DE SEGURANÇA & RETRIEVAL ---
MAX_SAFE_TOKENS = 80000     
MAX_SAFE_CHARS = MAX_SAFE_TOKENS * 4 
MAX_RETRIEVED_DOCS = 5      
CONTEXT_WINDOW_SIZE = 2      

LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.3            
BASE_URL = "https://api.deepseek.com"

# --- SETUP SUPABASE (MEMÓRIA DINÂMICA) ---
# Tenta pegar dos secrets (Cloud) ou variáveis de ambiente (Local)
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

# ============================================================================
# 2. ALGORITMO BM25 (CRUCIAL ESTAR AQUI PARA O PICKLE FUNCIONAR)
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

# ============================================================================
# 3. GESTÃO DE CONEXÕES E DADOS
# ============================================================================

@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Erro na conexão Supabase: {e}")
        return None

@st.cache_resource
def get_llm_client():
    # Tenta pegar a chave do Secrets ou do .env
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("Chave da API DeepSeek não encontrada.")
        return None
    return OpenAI(base_url=BASE_URL, api_key=api_key)

def load_memory_from_disk() -> Tuple[Optional[pd.DataFrame], Optional[SimpleBM25]]:
    """Carrega a memória estática (Git)."""
    if os.path.exists(DB_FILE) and os.path.exists(INDEX_FILE):
        try:
            df = pd.read_parquet(DB_FILE)
            with open(INDEX_FILE, 'rb') as f:
                bm25 = pickle.load(f)
            return df, bm25
        except Exception as e:
            st.warning(f"Erro ao carregar memória do disco: {e}")
            return None, None
    return None, None

# ============================================================================
# 4. GESTÃO DE CHAT (SUPABASE)
# ============================================================================

def load_chat_history() -> Dict:
    """Baixa o histórico do Supabase."""
    supabase = get_supabase_client()
    if not supabase: return {}
    
    try:
        response = supabase.table("chat_sessions").select("*").order("last_updated", desc=True).execute()
        history = {}
        for row in response.data:
            history[row['session_id']] = {
                "title": row['title'],
                "timestamp": row['last_updated'],
                "messages": row['messages']
            }
        return history
    except Exception as e:
        # Silencia erro se for só falta de tabela na primeira vez
        print(f"Aviso Supabase: {e}")
        return {}

def save_chat_session_remote(session_id: str, session_data: Dict):
    """Salva/Atualiza sessão no Supabase."""
    supabase = get_supabase_client()
    if not supabase: return

    try:
        data = {
            "session_id": session_id,
            "title": session_data["title"],
            "messages": session_data["messages"],
            "last_updated": datetime.now().isoformat()
        }
        supabase.table("chat_sessions").upsert(data).execute()
    except Exception as e:
        print(f"Erro ao salvar remoto: {e}")

def create_new_chat_session():
    new_id = str(uuid.uuid4())
    new_data = {
        "title": "Nova Conversa",
        "timestamp": datetime.now().isoformat(),
        "messages": []
    }
    st.session_state.all_chats[new_id] = new_data
    st.session_state.active_session_id = new_id
    save_chat_session_remote(new_id, new_data)

def delete_chat_session(session_id):
    supabase = get_supabase_client()
    if session_id in st.session_state.all_chats:
        del st.session_state.all_chats[session_id]
    
    if supabase:
        try:
            supabase.table("chat_sessions").delete().eq("session_id", session_id).execute()
        except Exception as e:
            st.error(f"Erro ao deletar da nuvem: {e}")

    # Redirecionamento inteligente
    remaining_ids = list(st.session_state.all_chats.keys())
    if remaining_ids:
        # Ordena para pegar o mais recente
        sorted_ids = sorted(st.session_state.all_chats.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        st.session_state.active_session_id = sorted_ids[0][0]
    else:
        create_new_chat_session()
    
    st.rerun()

# ============================================================================
# 5. MOTOR DE BUSCA E GERAÇÃO (CORE)
# ============================================================================

def retrieve_context(query: str, df: pd.DataFrame, bm25: SimpleBM25) -> str:
    if df is None or bm25 is None: return ""
    
    # Validação de Segurança
    if len(df) != bm25.corpus_size:
        return ""

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
        3. Use a personalidade do Thiago (Skin in the game, longo prazo)
        4. Seja extremamente detalhista, profundo e abrangente no máximo que você puder.
        5. Use apenas o CONTEXTO mais recente fornecido (Não use o conhecimento geral de treinamento do modelo) para responder.
        6. Sempre referencie nas suas respostas, o video mais recente utilizado para a mesma.
        7. Se o contexto for cortado, use o que tem disponível.
        8. Seja visionário, prático e conselheiro ou coach financeiro.
        9. Incorpore a essência intrínseca da alma do Thiago Nigro, use seu jeito de falar, suas gírias e sua personalidade única. Copie-o, Imite-o.
    """
    
    full_prompt = f"CONTEXTO RECUPERADO:\n{context}\n\nPERGUNTA DO PRIMO:\n{query}"
    client = get_llm_client()
    if not client: return None

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_persona}, {"role": "user", "content": full_prompt}],
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=8000
    )
    return stream

# ============================================================================
# 6. UI PRINCIPAL
# ============================================================================

def main():
    st.set_page_config(
        page_title="Primo.AI | Gêmeo Digital", 
        page_icon=LOGO_PATH2, 
        layout="wide"
    )
    
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: #f0f2f6; } 
            .stChatMessage { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; }
            [data-testid="stSidebarNav"] { display: none; }
            div[data-testid="stSidebar"] button {
                text-align: left;
                width: 100%;
                border-radius: 8px;
                margin-bottom: 4px;
                padding: 8px 12px;
                background-color: #262730;
                border: 1px solid transparent;
            }
            div[data-testid="stSidebar"] button:hover {
                border-color: #fca311;
                background-color: #31333f;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- LOADING INICIAL ---
    if "db" not in st.session_state:
        # Carrega Memória Estática (Git)
        df_disk, bm25_disk = load_memory_from_disk()
        st.session_state.db = df_disk
        st.session_state.bm25 = bm25_disk
    
    if "all_chats" not in st.session_state:
        # Carrega Memória Dinâmica (Supabase)
        st.session_state.all_chats = load_chat_history()
    
    if "active_session_id" not in st.session_state:
        if st.session_state.all_chats:
            sorted_chats = sorted(st.session_state.all_chats.items(), key=lambda x: x[1]['timestamp'], reverse=True)
            st.session_state.active_session_id = sorted_chats[0][0]
        else:
            create_new_chat_session()

    # --- SIDEBAR ---
    
    with st.sidebar:
        c1, c2 = st.columns([1, 4])
        with c1:
            try: st.image(LOGO_PATH, width=50)
            except: st.write("🤖")
        with c2:
            st.title("Primo.AI")
            st.caption("Gêmeo Digital do Thiago Nigro | Desenvolvido por Gabriel Estrela")
        
        if st.button("➕ Nova Conversa", type="primary", use_container_width=True):
            create_new_chat_session()
            st.rerun()
            
        st.markdown("---")
        st.caption("HISTÓRICO")

        # Lista de conversas ordenadas
        sorted_chats = sorted(
            st.session_state.all_chats.items(), 
            key=lambda x: x[1].get('timestamp', ''), 
            reverse=True
        )

        for cid, cdata in sorted_chats:
            is_active = (cid == st.session_state.active_session_id)
            title = cdata.get('title', 'Conversa sem título')
            btn_label = f"{'🟢' if is_active else '💬'} {title}"
            
            if st.button(btn_label, key=f"btn_{cid}"):
                st.session_state.active_session_id = cid
                st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Apagar Chat Atual"):
            delete_chat_session(st.session_state.active_session_id)

    # --- CHAT AREA ---
    if st.session_state.active_session_id not in st.session_state.all_chats:
        create_new_chat_session()
        st.rerun()

    current_id = st.session_state.active_session_id
    current_chat = st.session_state.all_chats[current_id]
    messages = current_chat["messages"]

    # Renderiza mensagens anteriores
    for msg in messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Pergunte ao Primo..."):
        # Se não tiver memória carregada
        if st.session_state.db is None:
            st.error("⚠️ Memória não encontrada. Verifique se a pasta 'primo_memory' está no GitHub.")
            st.stop()

        # 1. Usuário
        messages.append({"role": "user", "content": prompt})
        st.session_state.all_chats[current_id]["messages"] = messages
        
        # Auto-título na primeira mensagem
        if len(messages) == 1:
            st.session_state.all_chats[current_id]["title"] = (prompt[:30] + "...") if len(prompt) > 30 else prompt
        
        st.session_state.all_chats[current_id]["timestamp"] = datetime.now().isoformat()
        save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
        
        with st.chat_message("user"): st.markdown(prompt)

        # 2. Assistente
        with st.chat_message("assistant"):
            resp_container = st.empty()
            with st.spinner("Consultando livros e vídeos..."):
                context = retrieve_context(prompt, st.session_state.db, st.session_state.bm25)
            
            if not context:
                msg_fail = "Primo, não achei nada específico sobre isso nos meus arquivos. Tem certeza que já me ensinou?"
                resp_container.markdown(msg_fail)
                messages.append({"role": "assistant", "content": msg_fail})
                st.session_state.all_chats[current_id]["messages"] = messages
                save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
            else:
                full_res = ""
                try:
                    stream = generate_response(prompt, context)
                    if stream:
                        for chunk in stream:
                            content = chunk.choices[0].delta.content or ""
                            full_res += content
                            resp_container.markdown(full_res + "▌")
                        
                        resp_container.markdown(full_res)
                        messages.append({"role": "assistant", "content": full_res})
                        st.session_state.all_chats[current_id]["messages"] = messages
                        save_chat_session_remote(current_id, st.session_state.all_chats[current_id])
                        
                        if len(messages) <= 2: st.rerun()
                except Exception as e:
                    st.error(f"Erro na API: {e}")

if __name__ == "__main__":
    main()