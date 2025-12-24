# app.py
# ----------------------------------------------------------------------------
# ARQUITETURA: RAG Local (Git) + DeepSeek + Sessão Ephemeral (100% Privada)
# AUTOR: Ph.D. Assistant & User
# VERSÃO: 9.1 (Privacy Edition - Sem Banco de Dados Externo)
# ----------------------------------------------------------------------------

import os
import math
import re
import pickle
import logging
import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# 1. CONFIGURAÇÃO & HIPERPARÂMETROS
# ============================================================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CAMINHOS DE PERSISTÊNCIA (MEMÓRIA DE CONHECIMENTO) ---
# A pasta 'primo_memory' deve estar na raiz do seu repositório Git
MEMORY_DIR = "primo_memory"
DB_FILE = os.path.join(MEMORY_DIR, "knowledge_base.parquet")
INDEX_FILE = os.path.join(MEMORY_DIR, "bm25_index.pkl")

# --- BRANDING ---
# Certifique-se que as imagens estão na raiz
LOGO_PATH = "Primo_LOGO-removebg-preview.png" 
LOGO_PATH2 = "Logo_primo.png"

# --- TUNING DE RETRIEVAL (Ajuste Fino da Busca) ---
MAX_SAFE_TOKENS = 80000     
MAX_SAFE_CHARS = MAX_SAFE_TOKENS * 4 
MAX_RETRIEVED_DOCS = 5      
CONTEXT_WINDOW_SIZE = 2      

# --- LLM CONFIG (DeepSeek) ---
LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.3            
BASE_URL = "https://api.deepseek.com"

# ============================================================================
# 2. ALGORITMO BM25 (MOTOR DE BUSCA)
# ============================================================================
# Esta classe é necessária para ler o arquivo .pkl gerado anteriormente

class SimpleBM25:
    """Implementação leve do BM25. Otimizada para ser carregada via Pickle."""
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.k1 = 1.5
        self.b = 0.75
        # Stopwords em Português para limpeza
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
def get_llm_client():
    """Conecta à API do DeepSeek usando secrets do Streamlit ou .env"""
    api_key = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("⚠️ Chave da API DeepSeek não encontrada. Adicione ao .env ou Secrets.")
        return None
    return OpenAI(base_url=BASE_URL, api_key=api_key)

@st.cache_resource
def load_memory_from_disk() -> Tuple[Optional[pd.DataFrame], Optional[SimpleBM25]]:
    """Carrega a memória estática (Transcrições) do disco/Git."""
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
# 4. MOTOR DE BUSCA E GERAÇÃO (CORE)
# ============================================================================

def retrieve_context(query: str, df: pd.DataFrame, bm25: SimpleBM25) -> str:
    """Busca os trechos mais relevantes nas transcrições."""
    if df is None or bm25 is None: return ""
    
    # Validação de Segurança e Integridade
    if len(df) != bm25.corpus_size:
        return ""

    scores = bm25.get_scores(query)
    # Seleciona os índices com maior pontuação
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:MAX_RETRIEVED_DOCS]
    # Filtra apenas o que tiver relevância mínima (> 1.0)
    top_indices = [i for i in top_indices if scores[i] > 1.0]

    if not top_indices: return ""

    expanded_indices = set()
    for idx in top_indices:
        # Pega janelas de contexto (antes e depois do trecho encontrado)
        start = max(0, idx - CONTEXT_WINDOW_SIZE)
        end = min(len(df), idx + CONTEXT_WINDOW_SIZE + 1)
        original_source = df.iloc[idx]['source_title']
        for i in range(start, end):
            # Garante que não misture vídeos diferentes
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
    """Gera a resposta usando o DeepSeek com a persona do Primo."""
    
    system_persona = """
        Você é o Gêmeo Digital do Thiago Nigro.
        DIRETRIZES:
        1. Seja detalhista e use o CONTEXTO fornecido.
        2. Cite os vídeos/fontes do contexto recuperado.
        3. Use a personalidade e as gírias do Thiago (foco no longo prazo).
        4. Seja extremamente detalhista, profundo e abrangente no máximo que você puder.
        5. Use apenas o CONTEXTO mais recente fornecido (Não use o conhecimento geral de treinamento do modelo se contradizer o contexto).
        6. Sempre referencie nas suas respostas o vídeo mais recente utilizado.
        7. Se o contexto for cortado, use o que tem disponível.
        8. Seja visionário, prático e aja como um conselheiro/coach financeiro sênior.
        9. Incorpore a essência intrínseca da alma do Thiago Nigro: use seu jeito de falar, suas gírias ("Primo", "Sócio"), e sua personalidade única. Imite-o perfeitamente.
"Agora tome uma respiração profunda , respire fundo,fique calmo e responda como o Thiago Nigro faria."
    """
    
    full_prompt = f"CONTEXTO RECUPERADO:\n{context}\n\nPERGUNTA DO PRIMO:\n{query}"
    client = get_llm_client()
    if not client: return None

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_persona}, {"role": "user", "content": full_prompt}],
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=8000 # Máximo permitido
    )
    return stream

# ============================================================================
# 5. UI PRINCIPAL (STREAMLIT)
# ============================================================================

def main():
    st.set_page_config(
        page_title="Primo.AI | Gêmeo Digital", 
        page_icon=LOGO_PATH2, 
        layout="wide"
    )
    
    # CSS Customizado para Dark Mode e Chat
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: #f0f2f6; } 
            .stChatMessage { background-color: #1f2937; border: 1px solid #374151; border-radius: 12px; }
            /* Ocultar menu padrão do Streamlit */
            [data-testid="stSidebarNav"] { display: none; }
            div[data-testid="stSidebar"] { background-color: #111; }
        </style>
    """, unsafe_allow_html=True)

    # --- LOADING MEMÓRIA (RAG) ---
    if "db" not in st.session_state:
        # Carrega a memória estática contendo as transcrições
        with st.spinner("Carregando cérebro do Primo..."):
            df_disk, bm25_disk = load_memory_from_disk()
            st.session_state.db = df_disk
            st.session_state.bm25 = bm25_disk
    
    # --- INICIALIZAÇÃO DA SESSÃO ---
    # messages só existem enquanto a aba estiver aberta
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR (CONTROLES) ---
    with st.sidebar:
        c1, c2 = st.columns([1, 4])
        with c1:
            try: st.image(LOGO_PATH, width=50)
            except: st.write("🧠")
        with c2:
            st.title("Primo.AI")
            st.caption("Gêmeo Digital | Desenvolvido por Gabriel Estrela")
        
        st.markdown("---")
        
        
        st.markdown("### Ações")
        if st.button("🧹 Limpar Chat e Começar de Novo", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

    # --- ÁREA DE CHAT ---

    # 1. Renderiza mensagens anteriores (apenas desta sessão)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # 2. Input do Usuário
    if prompt := st.chat_input("Pergunte ao Primo sobre investimentos, negócios ou mentalidade..."):
        
        # Verificação de integridade da memória antes de prosseguir
        if st.session_state.db is None:
            st.error("⚠️ Memória não encontrada. Verifique se a pasta 'primo_memory' com os arquivos .parquet e .pkl está no diretório correto.")
            st.stop()

        # Adiciona pergunta do usuário à tela e estado
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        # 3. Resposta do Assistente
        with st.chat_message("assistant"):
            resp_container = st.empty()
            
            # Retrieval (Busca nas transcrições locais)
            with st.spinner("Consultando biblioteca mental do Primo..."):
                context = retrieve_context(prompt, st.session_state.db, st.session_state.bm25)
            
            # Lógica de Falha ou Sucesso
            if not context:
                msg_fail = "E aí, primo! Tudo bem com você?Procurei aqui em todos os meus vídeos e livros, mas não achei nada específico sobre isso no meu contexto atual. Você tem certeza que eu já falei sobre isso ou se trata de uma pergunta solta?"
                resp_container.markdown(msg_fail)
                st.session_state.messages.append({"role": "assistant", "content": msg_fail})
            else:
                full_res = ""
                try:
                    # Chama LLM com Streaming
                    stream = generate_response(prompt, context)
                    if stream:
                        for chunk in stream:
                            content = chunk.choices[0].delta.content or ""
                            full_res += content
                            # Efeito de digitação
                            resp_container.markdown(full_res + "▌")
                        
                        # Renderiza final
                        resp_container.markdown(full_res)
                        st.session_state.messages.append({"role": "assistant", "content": full_res})
                except Exception as e:
                    st.error(f"Erro ao conectar com o cérebro digital: {e}")

if __name__ == "__main__":
    main()