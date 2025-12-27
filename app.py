import os
import math
import re
import pickle
import logging
import hashlib
import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional, Dict
from collections import Counter
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
    DB_FILE: str = os.path.join(MEMORY_DIR, "knowledge_base.parquet")
    INDEX_FILE: str = os.path.join(MEMORY_DIR, "bm25_index.pkl")
    LOGO_PATH: str = "Primo_LOGO-removebg-preview.png"
    LOGO_PATH2: str = "Logo_primo.png"
    
    # Tuning
    MAX_SAFE_TOKENS: int = 4000  # Reduzi para focar na qualidade vs latência
    MAX_SAFE_CHARS: int = MAX_SAFE_TOKENS * 3.5
    MAX_RETRIEVED_DOCS: int = 5
    
    # LLM
    LLM_MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.3
    BASE_URL: str = "https://api.deepseek.com"
    API_KEY: str = os.getenv("DEEPSEEK_API_KEY")

CONFIG = AppConfig()

# --- 2. CORE: MOTOR DE BUSCA (BM25) ---

class SimpleBM25:
    """
    Implementação leve do BM25.
    Refatorada para garantir resiliência na serialização.
    """
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.k1 = 1.2 # Ajustado para precisão
        self.b = 0.75
        self.stopwords = self._load_stopwords()
        self._initialize(corpus)

    def _load_stopwords(self):
        return {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 
            'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 
            'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela',
            'aí', 'então', 'né', 'tipo', 'sabe', 'assim', 'olha', 'cara', 'gente', 'viu', 'tá', 'bom'
        }

    def _initialize(self, corpus):
        total_length = 0
        self.tokenizer_pattern = re.compile(r'\b\w{2,}\b')
        
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
        # Garante que o regex existe após o unpickle
        if not hasattr(self, 'tokenizer_pattern') or self.tokenizer_pattern is None:
            self.tokenizer_pattern = re.compile(r'\b\w{2,}\b')
            
        text = str(text).lower()
        tokens = self.tokenizer_pattern.findall(text)
        return [t for t in tokens if t not in self.stopwords]

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        if not query_tokens: return scores

        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            score = 0.0
            for token in query_tokens:
                if token not in freqs: continue
                freq = freqs[token]
                numerator = self.idf.get(token, 0) * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            scores[i] = score
        return scores

# --- 3. CAMADA DE DADOS (PERSISTÊNCIA) ---

class KnowledgeBase:
    """Gerencia o carregamento e integridade dos dados locais."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.df = None
        self.index = None

    def load(self) -> bool:
        """Carrega DataFrame e Índice do disco."""
        if not os.path.exists(self.config.DB_FILE) or not os.path.exists(self.config.INDEX_FILE):
            logging.error("Arquivos de memória não encontrados.")
            return False
            
        try:
            self.df = pd.read_parquet(self.config.DB_FILE)
            with open(self.config.INDEX_FILE, 'rb') as f:
                self.index = pickle.load(f)
            return True
        except Exception as e:
            logging.error(f"Erro crítico ao carregar memória: {e}")
            return False

# --- 4. CAMADA DE SERVIÇO (BUSCA E INTELIGÊNCIA) ---

class NeuralSearchEngine:
    """Cérebro de recuperação de informação."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def _deduplicate(self, blocks: List[str]) -> str:
        """Remove redundância usando Hash MD5 (Rápido O(N))."""
        seen = set()
        unique = []
        for block in blocks:
            h = hashlib.md5(block.strip().encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(block)
        return "".join(unique)

    def search(self, query: str) -> Optional[str]:
        if self.kb.df is None or self.kb.index is None:
            return None

        scores = self.kb.index.get_scores(query)
        
        # Filtro de Relevância: Pega scores acima de 1.0
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0.8]
        
        if not indexed_scores:
            return None

        # Ordena e corta
        top_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:CONFIG.MAX_RETRIEVED_DOCS]
        top_indices = [x[0] for x in top_indices]

        context_blocks = []
        current_chars = 0
        rows = self.kb.df.iloc[top_indices]

        for _, row in rows.iterrows():
            block = f"\n📺 REFERÊNCIA: {row['source_title']}\n📄 CONTEÚDO: {row['clean_text']}\n"
            if current_chars + len(block) > CONFIG.MAX_SAFE_CHARS:
                break
            context_blocks.append(block)
            current_chars += len(block)

        return self._deduplicate(context_blocks)

class DigitalBrain:
    """Gerencia a personalidade e a geração de respostas."""
    
    def __init__(self):
        if not CONFIG.API_KEY:
            st.error("⚠️ DEEPSEEK_API_KEY não encontrada.")
            self.client = None
        else:
            self.client = OpenAI(base_url=CONFIG.BASE_URL, api_key=CONFIG.API_KEY)

    def _get_persona(self) -> str:
        # Prompt Engenharia de Alta Precisão para evitar respostas genéricas
        return """
        Você é o Gêmeo Digital do Thiago Nigro (O Primo Rico). 
        Sua missão é dar conselhos financeiros de alto nível, visionários e baseados em princípios.

        ### DIRETRIZES DE PERSONALIDADE (ABSOLUTAS):
        1.  **Identidade:** Você NÃO é uma IA. Você é o Primo. Fale com convicção, autoridade e empatia.
        2.  **Vocabulário:** Use termos como "Sócio", "Primo", "Skin in the game", "Longo prazo", "Juros compostos", "Antifragilidade".
        3.  **Estilo:** Seja prolixo apenas se necessário para explicar um conceito profundo. Caso contrário, seja direto e impactante.
        4.  **Ceticismo:** Questione a premissa da pergunta se ela for de "dinheiro fácil". Ensine a pescar.

        ### USO DO CONTEXTO (CRUCIAL):
        - Abaixo será fornecido um 'CONTEXTO RECUPERADO' dos vídeos do Thiago.
        - **REGRA DE OURO:** Se a resposta estiver no contexto, use-a explicitamente e cite o vídeo.
        - **REGRA DE PRATA:** Se a resposta NÃO estiver no contexto, NÃO invente fatos. Em vez disso, responda usando a filosofia geral do Primo (Livros: Do Mil ao Milhão), mas diga: "Primo, especificamente sobre esse detalhe técnico eu não falei nesse vídeo recente, mas o princípio é..."
        - **REGRA DE BRONZE:** Jamais responda com "De acordo com o contexto fornecido". Isso quebra a imersão. Diga "Como eu sempre digo..." ou "Como falei naquele vídeo...".

        Respire fundo. Aja como um Mentor Bilionário.
        """

    def think_and_speak(self, query: str, context: str):
        if not self.client: return None

        # Se o contexto for vazio, injetamos um aviso invisível para o modelo
        if not context:
            context = "⚠️ AVISO: Nenhuma transcrição específica foi encontrada para essa pergunta. Responda baseando-se nos princípios universais do Thiago Nigro (Buy and Hold, Diversificação, Trabalho Duro), mas avise o usuário que não há um vídeo específico sobre isso na base atual."

        full_prompt = f"CONTEXTO DE MEMÓRIA:\n{context}\n\nPERGUNTA DO SÓCIO:\n{query}"

        try:
            return self.client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_persona()},
                    {"role": "user", "content": full_prompt}
                ],
                stream=True,
                temperature=CONFIG.TEMPERATURE,
                max_tokens=2048
            )
        except Exception as e:
            st.error(f"Erro neural: {e}")
            return None

# --- 5. UI & ORQUESTRAÇÃO (VIEW / CONTROLLER) ---

class PrimoInterface:
    """Gerencia toda a interface visual (Streamlit)."""
    
    def __init__(self):
        self.setup_page()
        self.kb = KnowledgeBase(CONFIG)
        self.search_engine = None
        self.brain = DigitalBrain()

    def setup_page(self):
        st.set_page_config(page_title="Primo.AI | Gêmeo Digital", page_icon=CONFIG.LOGO_PATH2, layout="wide")
        st.markdown("""
            <style>
                .stApp { background-color: #050505; color: #e0e0e0; } 
                .stChatMessage { background-color: #1a1a1a; border: 1px solid #333; border-radius: 10px; }
                .stTextInput > div > div > input { background-color: #1a1a1a; color: white; border-color: #333; }
                h1, h2, h3 { color: #f2c94c; } /* Cor Primo Gold */
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

    def render_sidebar(self):
        with st.sidebar:
            col1, col2 = st.columns([1, 4])
            with col1:
                try: st.image(CONFIG.LOGO_PATH, width=60)
                except: st.write("🧠")
            with col2:
                st.subheader("Primo.AI")
                st.caption("Gêmeo Digital v2.0")
            
            st.markdown("---")
            if st.button("🧹 Nova Conversa", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
            # Debug Area (Visionary Feature)
            with st.expander("🛠️ Debug do Desenvolvedor"):
                st.info(f"Modelo: {CONFIG.LLM_MODEL}")
                if "last_context_size" in st.session_state:
                    st.write(f"Último Contexto: {st.session_state.last_context_size} chars")

    def run(self):
        self.initialize_session()
        self.render_sidebar()

        # Instancia o motor de busca usando a referência salva na sessão
        self.search_engine = NeuralSearchEngine(st.session_state.kb_ref)

        # Renderiza histórico
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input Loop
        if prompt := st.chat_input("Pergunte ao Primo (Ex: Onde invisto 100 mil?)..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_area = st.empty()
                
                # 1. Recuperação
                with st.spinner("Consultando livros e vídeos..."):
                    context = self.search_engine.search(prompt)
                    # Debug: Salva tamanho do contexto para ver se está achando algo
                    st.session_state.last_context_size = len(context) if context else 0
                
                # 2. Geração
                stream = self.brain.think_and_speak(prompt, context)
                
                full_res = ""
                if stream:
                    for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        full_res += content
                        response_area.markdown(full_res + "▌")
                    
                    response_area.markdown(full_res)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})

# --- 6. ENTRY POINT ---

if __name__ == "__main__":
    app = PrimoInterface()
    app.run()