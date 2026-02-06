import os
import re
import logging
import time # Adicionado para sleep
import streamlit as st
import pandas as pd
import json
import ast 
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass

# --- 1. CONFIGURAÇÃO & SINGLETONS ---

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class AppConfig:
    """Configurações centralizadas da aplicação."""
    MEMORY_DIR: str = "primo_memory"
    CHROMA_DB_DIR: str = os.path.join(MEMORY_DIR, "chroma_db")
    DB_FILE: str = os.path.join(MEMORY_DIR, "knowledge_base.parquet")
    LOGO_PATH: str = "Primo_LOGO-removebg-preview.png"
    LOGO_PATH2: str = "Logo_primo.png"
    
    # Tuning RAG
    MAX_RETRIEVED_DOCS: int = 5
    CHUNK_SIZE: int = 1000  # Tamanho do chunk de texto
    CHUNK_OVERLAP: int = 200 # Sobreposição para contexto
    
    # LLM
    LLM_MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.6 
    BASE_URL: str = "https://api.deepseek.com"
    API_KEY: str = os.getenv("DEEPSEEK_API_KEY")

CONFIG = AppConfig()

# --- 2. CORE: MEMÓRIA VETORIAL (CHROMADB) ---

class VectorMemory:
    """Gerencia o banco de dados vetorial para busca semântica."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        # Inicializa o cliente Chroma Persistente
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        
        # Usa o modelo de embedding padrão (all-MiniLM-L6-v2) que é leve e eficiente
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="primo_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Similaridade de cosseno
        )

    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        """Adiciona textos vetorizados ao banco."""
        if not texts:
            return
            
        # Processamento em lotes para evitar estouro de memória
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.upsert(
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        logging.info(f"💾 {len(texts)} chunks indexados no ChromaDB.")

    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        """Realiza a busca semântica."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Chroma retorna lista de listas (uma para cada query). Pegamos a primeira.
        if results and results['documents']:
            return results['documents'][0]
        return []

    def count(self):
        return self.collection.count()
        
    def get_existing_source_ids(self) -> List[str]:
        """Recupera IDs de vídeos já indexados para evitar duplicatas."""
        try:
            # Chroma get sem argumentos retorna tudo (cuidado com volume)
            # Para otimizar, pegamos apenas metadados
            data = self.collection.get(include=['metadatas'])
            metas = data.get('metadatas', [])
            
            # Extrai 'source_info' ou ID do vídeo dos metadados se existir
            # Como salvamos no chunk_id como "{source_info}_{i}", podemos usar regex ou metadata
            # Na implementação anterior, 'source' no metadata era o Título. 
            # O ideal seria ter guardado o Video ID exatamente.
            # Workaround: Vamos tentar inferir dos metadados existentes ou mudar a indexação futura.
            
            # Melhor abordagem futura: Salvar 'video_id' no metadata.
            # Por agora, para não quebrar compatibilidade, vamos assumir que não temos ID fácil 
            # e retornar lista vazia para forçar check ou implementar extração melhor se possível.
            
            # Se salvamos chunk_id como "VIDEO_ID_CHUNK_INDEX", podemos dar split.
            ids = data.get('ids', [])
            video_ids = set()
            for chunk_id in ids:
                if "_" in chunk_id:
                     # Assume formato: ID_DO_VIDEO_0
                     # Cuidado: ID do YouTube também pode ter underscore? Sim.
                     # Vamos pegar tudo até o último underscore
                     vid_id = chunk_id.rsplit('_', 1)[0]
                     video_ids.add(vid_id)
            return list(video_ids)
        except:
             return []

# --- 3. CAMADA DE DADOS E PROCESSAMENTO ---

class KnowledgeBase:
    """Gerencia processamento de dados e chunking."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.memory = VectorMemory(config)

    def sync_new_videos(self):
        """Chama o scraper para buscar apenas vídeos novos."""
        try:
            # Import dinâmico para evitar erro circular ou de dependência se script faltar
            from Transcript_Channel import update_channel_data
        except ImportError:
            logging.error("Script Transcript_Channel.py não encontrado.")
            return 0 

        existing_ids = self.memory.get_existing_source_ids()
        logging.info(f"🔎 Analisando diff contra {len(existing_ids)} vídeos já existentes...")
        
        new_data = update_channel_data(existing_ids)
        
        if not new_data:
            return 0
            
        # Converte para DataFrame para reaproveitar lógica de indexação
        df_new = pd.DataFrame(new_data)
        
        # Indexa no Chroma
        self.process_and_index(df_new)
        return len(new_data)

    def _clean_text(self, text):
        """Limpeza básica de texto."""
        if pd.isna(text) or text == "": return ""
        text = str(text).replace("\n", " ").strip()
        return re.sub(r'\s+', ' ', text)

    def _chunk_text(self, text: str, source_info: str) -> List[tuple]:
        """Divide o texto em chunks com sobreposição."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.config.CHUNK_SIZE - self.config.CHUNK_OVERLAP):
            chunk_words = words[i : i + self.config.CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            
            # ID único para o chunk
            chunk_id = f"{source_info}_{i}"
            chunks.append((chunk_id, chunk_text))
            
        return chunks

    def process_and_index(self, df: pd.DataFrame):
        """Processa o DataFrame, cria chunks e indexa no ChromaDB."""
        all_docs = []
        all_metas = []
        all_ids = []

        logging.info("⚙️ Iniciando processamento e chunking...")
        
        for idx, row in df.iterrows():
            # Extração robusta do texto (adaptada do código original)
            raw_text = row.get('text', row.get('clean_text', ''))
            
            # Se for JSON string, tenta parsear
            if isinstance(raw_text, str) and (raw_text.startswith('[') or raw_text.startswith('{')):
                try:
                    parsed = json.loads(raw_text.replace("'", '"'))
                    if isinstance(parsed, list):
                        raw_text = " ".join([p.get('text', '') for p in parsed if 'text' in p])
                except:
                    pass

            clean_text = self._clean_text(raw_text)
            if not clean_text or len(clean_text) < 50: continue

            source = row.get('source_title', f'doc_{idx}')
            chunks = self._chunk_text(clean_text, str(idx))

            for chunk_id, chunk_content in chunks:
                all_ids.append(chunk_id)
                all_docs.append(chunk_content)
                all_metas.append({"source": source})

        if all_docs:
            self.memory.add_texts(all_docs, all_metas, all_ids)
            return True
        return False

    def load(self) -> bool:
        # Verifica se já temos dados indexados
        if self.memory.count() > 0:
            logging.info(f"✅ ChromaDB já carregado com {self.memory.count()} chunks.")
            return True

        # Se não, carrega do Parquet/JSON
        if not os.path.exists(self.config.DB_FILE):
             # Tenta achar JSON se não tiver Parquet
             json_files = [f for f in os.listdir(".") if f.endswith(".json") and "transcricoes" in f]
             if json_files:
                 try:
                    df = pd.read_json(json_files[0])
                    return self.process_and_index(df)
                 except Exception as e:
                     logging.error(f"Erro ao ler JSON: {e}")
                     return False
             return False
            
        try:
            df = pd.read_parquet(self.config.DB_FILE)
            return self.process_and_index(df)
        except Exception as e:
            logging.error(f"Erro ao ler Parquet: {e}")
            return False

# --- 4. ENGINE DE BUSCA ---

class NeuralSearchEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def search(self, query: str) -> Optional[str]:
        # Busca Semântica via ChromaDB
        results = self.kb.memory.query(query, n_results=CONFIG.MAX_RETRIEVED_DOCS)
        
        if not results:
            return None

        # Monta o contexto
        context_blocks = []
        for text in results:
            context_blocks.append(f"- {text}")
            
        return "\n\n".join(context_blocks)

# --- 5. CÉREBRO DIGITAL & SEGURANÇA ---

class DigitalBrain:
    def __init__(self):
        if not CONFIG.API_KEY:
            st.error("⚠️ KEY não encontrada. Configure o .env")
            self.client = None
        else:
            self.client = OpenAI(base_url=CONFIG.BASE_URL, api_key=CONFIG.API_KEY)

    def _sanitize_input(self, text: str) -> str:
        """Remove caracteres perigosos para evitar injeção básica."""
        return re.sub(r'[{}@]', '', text[:1000]) # Limita tamanho e remove chars especiais de template

    def _get_persona(self) -> str:
        return """
        Você é o Gêmeo Digital do Thiago Nigro (O Primo Rico).
        Seu objetivo é dar conselhos financeiros, de carreira e negócios baseados EXCLUSIVAMENTE no contexto fornecido.
        
        DIRETRIZES:
        1.  Use o tom direto, prático e motivador do Thiago.
        2.  Se o contexto não tiver a resposta, DIGA que não sabe com base nos vídeos atuais. NÃO invente.
        3.  Cite princípios como "Skin in the Game", "Antifragilidade", "Diversificação".
        4.  Responda sempre em Português do Brasil.
        """

    def think_and_speak(self, query: str, context: str):
        if not self.client: return None

        safe_query = self._sanitize_input(query)
        
        if not context:
            context = "Nenhuma informação específica encontrada na base de conhecimento."

        # Prompt estruturado
        messages = [
            {"role": "system", "content": self._get_persona()},
            {"role": "user", "content": f"CONTEXTO:\n{context}\n\nPERGUNTA DO USUÁRIO:\n{safe_query}"}
        ]

        try:
            return self.client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=messages,
                stream=True,
                temperature=CONFIG.TEMPERATURE,
            )
        except Exception as e:
            st.error(f"Erro na API LLM: {e}")
            return None

# --- 6. INTERFACE (STREAMLIT) ---

class PrimoInterface:
    def __init__(self):
        st.set_page_config(page_title="Primo.AI 2.0 | RAG Neural", page_icon="🧠", layout="wide")
        self.apply_styles()
        self.kb = KnowledgeBase(CONFIG)
        self.brain = DigitalBrain()

    def apply_styles(self):
        st.markdown("""
            <style>
                .stApp { background-color: #050505; color: #e0e0e0; }
                .stChatMessage { background-color: #1a1a1a; border: 1px solid #333; }
                h1, h2, h3 { color: #f2c94c; }
            </style>
        """, unsafe_allow_html=True)

    def initialize_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Carregamento Lazy da Memória
        if "knowledge_loaded" not in st.session_state:
            with st.spinner("🧠 Sintonizando frequência mental do Primo..."):
                success = self.kb.load()
                if success:
                    st.session_state.kb_ref = self.kb
                    st.session_state.knowledge_loaded = True
                else:
                    st.error("Falha crítica ao carregar cérebro digital.")
                    st.stop()

        # --- AUTO-SYNC (NOVO: Atualização Automática no Login) ---
        if "auto_sync_done" not in st.session_state:
            with st.status("🔄 Verificando novos vídeos no YouTube...", expanded=True) as status:
                st.write("Conectando ao canal do Primo...")
                count = self.kb.sync_new_videos()
                if count > 0:
                     status.update(label=f"✅ Base atualizada! {count} novos vídeos aprendidos.", state="complete", expanded=False)
                     time.sleep(1) # Breve pausa para o usuário ver
                else:
                     status.update(label="✅ Sistema já está atualizado.", state="complete", expanded=False)
            
            st.session_state.auto_sync_done = True
            if count > 0:
                 st.rerun() # Refresh para carregar os novos dados na memória ativa

    def render_sidebar(self):
        with st.sidebar:
            st.image(CONFIG.LOGO_PATH) if os.path.exists(CONFIG.LOGO_PATH) else st.write("🧠 Primo.AI")
            st.markdown("### Status do Sistema")
            
            # Status do ChromaDB
            try:
                doc_count = self.kb.memory.count()
                st.write(f"📚 Memória Vetorial: **{doc_count} chunks**")
            except:
                st.write("🔴 Memória Offline")

            # --- FORCE SYNC BUTTON (Backup) ---
            if st.button("🔄 Forçar Atualização"):
                 st.session_state.auto_sync_done = False # Reseta flag
                 st.rerun()

            if st.button("Recarregar/Reindexar Base"):
                st.cache_data.clear()
                # Simplesmente limpando a coleção para forçar reindexação seria ideal, 
                # mas aqui vamos apenas recarregar o obj
                st.session_state.knowledge_loaded = False # Força recarregamento
                st.rerun()
                
            if st.button("🗑️ Limpar Conversa"):
                st.session_state.messages = []
                st.rerun()

    def run(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Carregamento Inicial
        if "indices_loaded" not in st.session_state:
            with st.spinner("🚀 Atualizando córtex neural (Indexando Vetores)..."):
                self.kb.load()
                st.session_state.indices_loaded = True

        self.sidebar()
        self.search_engine = NeuralSearchEngine(self.kb)

        # Chat Loop
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Pergunte ao Primo..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # 1. Recuperação Neural
                context = self.search_engine.search(prompt)
                
                # Debug (Opcional)
                with st.expander("Ver Contexto Recuperado (RAG)"):
                    st.text(context or "Nada encontrado.")

                # 2. Geração
                stream = self.brain.think_and_speak(prompt, context)
                
                full_response = ""
                if stream:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    app = PrimoInterface()
    app.run()