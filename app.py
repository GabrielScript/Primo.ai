import os
import math
import re
import pickle
import logging
import hashlib
import streamlit as st
import pandas as pd
import json
import ast 
from typing import List, Tuple, Optional, Dict
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from dataclasses import dataclass
from datetime import datetime
import numpy as np

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
    LLM_MODEL: str = "xiaomi/mimo-v2-flash:free"
    TEMPERATURE: float = 0.6 # Aumentei levemente para o Xiaomi ser mais natural
    BASE_URL: str = "https://openrouter.ai/api/v1"
    MAX_TOKENS: int = 4096
    CONTEXT_LENGTH: int = 262144
    # Certifique-se de ter OPENROUTER_API_KEY no seu .env
    API_KEY: str = os.getenv("OPENROUTER_API_KEY")

CONFIG = AppConfig()

# --- 2. CORE: MOTOR DE BUSCA (BM25) ---

class SimpleBM25:
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
            'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à'
        }
        self.tokenizer_pattern = re.compile(r'\b\w{3,}\b') # Palavras com 3+ letras
        self._initialize(corpus)

    def _initialize(self, corpus):
        total_length = 0
        for document in corpus:
            tokens = self._tokenize(str(document))
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
        # Garantia de existência do pattern (essencial para Streamlit Cloud)
        if not hasattr(self, 'tokenizer_pattern') or self.tokenizer_pattern is None:
            self.tokenizer_pattern = re.compile(r'\b\w{3,}\b')
            
        text = str(text).lower()
        # Remove acentos básicos para aumentar a chance de match
        text = re.sub(r'[áàâã]', 'a', text)
        text = re.sub(r'[éèê]', 'e', text)
        text = re.sub(r'[íìî]', 'i', text)
        text = re.sub(r'[óòôõ]', 'o', text)
        text = re.sub(r'[úùû]', 'u', text)
        
        tokens = self.tokenizer_pattern.findall(text)
        return [t for t in tokens if t not in self.stopwords]

    def get_scores(self, query: str) -> List[float]:
        query_tokens = self._tokenize(query)
        # LOG DE DEBUG: Verifique isso no terminal!
        logging.info(f"🔎 Tokens extraídos da query: {query_tokens}")
        
        scores = [0.0] * self.corpus_size
        if not query_tokens: 
            return scores

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
    """Gerencia o carregamento, limpeza e indexação da memória."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.df = None
        self.index = None

    def _extract_text_from_json(self, raw_data):
        """
        Limpeza blindada para transcrições do YouTube.
        """
        if pd.isna(raw_data) or raw_data == "":
            return ""
            
        # Caso 1: Já é uma lista de dicionários (o pandas converteu sozinho)
        if isinstance(raw_data, list):
            data_list = raw_data
        
        # Caso 2: É uma string (JSON ou representação de lista Python)
        elif isinstance(raw_data, str):
            clean_str = raw_data.strip()
            # Tenta JSON padrão primeiro
            try:
                data_list = json.loads(clean_str.replace("'", '"')) # Tenta normalizar aspas simples
            except:
                # Fallback para ast.literal_eval (estrutura Python)
                try:
                    data_list = ast.literal_eval(clean_str)
                except:
                    # Se falhar tudo, assume que é texto puro (talvez já limpo)
                    return clean_str
        else:
            return str(raw_data)

        # Extração e Junção
        try:
            full_text = []
            for item in data_list:
                if isinstance(item, dict) and 'text' in item:
                    full_text.append(item['text'])
                elif isinstance(item, str):
                    full_text.append(item)
            
            return " ".join(full_text)
        except Exception as e:
            logging.error(f"Erro no parseamento final: {e}")
            return str(raw_data)

    def build_index(self):
        """Reconstrói o índice BM25 do zero usando o DataFrame limpo."""
        if self.df is None or self.df.empty:
            logging.error("Tentativa de indexar DataFrame vazio.")
            return False
            
        logging.info("♻️ Indexando conteúdo limpo...")
        try:
            # Indexa a coluna JÁ LIMPA
            if 'clean_text' not in self.df.columns:
                 # Fallback de segurança se a coluna não existir
                 logging.warning("Coluna 'clean_text' não encontrada. Criando agora...")
                 self.df['clean_text'] = self.df.iloc[:, 0].apply(self._extract_text_from_json)

            corpus = self.df['clean_text'].tolist()
            self.index = SimpleBM25(corpus)
            
            with open(self.config.INDEX_FILE, 'wb') as f:
                pickle.dump(self.index, f)
            logging.info("✅ Índice reconstruído e salvo.")
            return True
        except Exception as e:
            logging.error(f"Erro ao reconstruir índice: {e}")
            return False

    def load(self) -> bool:
        # 1. Carrega o Parquet
        if not os.path.exists(self.config.DB_FILE):
            return False
            
        try:
            self.df = pd.read_parquet(self.config.DB_FILE)
            
            # --- LIMPEZA AUTOMÁTICA ---
            # Verifica qual coluna tem os dados brutos. Geralmente é a primeira ou tem nome específico.
            # Aqui assumo que se 'clean_text' já existe, ok. Se não, cria.
            if 'clean_text' not in self.df.columns:
                 # Pega a primeira coluna de texto que achar se não souber o nome
                 target_col = self.df.columns[0] 
                 logging.info(f"🧹 Limpando dados brutos da coluna: {target_col}...")
                 self.df['clean_text'] = self.df[target_col].apply(self._extract_text_from_json)
            else:
                 # Se já existe, força a re-limpeza para garantir que o novo código seja aplicado
                 logging.info("🧹 Refazendo limpeza com novo algoritmo...")
                 self.df['clean_text'] = self.df['clean_text'].apply(self._extract_text_from_json)
            # -------------------------------------------
            
            logging.info(f"📚 Dados carregados e limpos: {len(self.df)} docs.")
        except Exception as e:
            logging.error(f"Erro crítico ao ler Parquet: {e}")
            return False

        # 2. Carrega ou Recria o Índice
        index_loaded = False
        if os.path.exists(self.config.INDEX_FILE):
            try:
                with open(self.config.INDEX_FILE, 'rb') as f:
                    self.index = pickle.load(f)
                index_loaded = True
            except:
                pass # Vai recriar se falhar
        
        if not index_loaded:
            return self.build_index()
            
        return True

    def load(self) -> bool:
        # 1. Carrega o Parquet
        if not os.path.exists(self.config.DB_FILE):
            return False
            
        try:
            self.df = pd.read_parquet(self.config.DB_FILE)
            
            # --- LIMPEZA AUTOMÁTICA (O Pulo do Gato) ---
            # Aplica a função de extração em TODAS as linhas
            logging.info("🧹 Limpando dados brutos de transcrição...")
            self.df['clean_text'] = self.df['clean_text'].apply(self._extract_text_from_json)
            # -------------------------------------------
            
            logging.info(f"📚 Dados carregados e limpos: {len(self.df)} docs.")
        except Exception as e:
            logging.error(f"Erro crítico ao ler Parquet: {e}")
            return False

        # 2. Carrega ou Recria o Índice
        index_loaded = False
        if os.path.exists(self.config.INDEX_FILE):
            try:
                with open(self.config.INDEX_FILE, 'rb') as f:
                    self.index = pickle.load(f)
                index_loaded = True
            except:
                pass # Vai recriar se falhar
        
        if not index_loaded:
            return self.build_index()
            
        return True

# --- 4. CAMADA DE SERVIÇO (BUSCA E INTELIGÊNCIA) ---

class NeuralSearchEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def search(self, query: str) -> Optional[str]:
        if self.kb.df is None or self.kb.index is None:
            logging.error("❌ KnowledgeBase ou Índice não carregados no NeuralSearchEngine.")
            return None

        # 1. Busca BM25
        scores = self.kb.index.get_scores(query)
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        
        top_indices = []
        if indexed_scores:
            top_indices = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:CONFIG.MAX_RETRIEVED_DOCS]
            logging.info(f"🎯 BM25 encontrou {len(top_indices)} documentos relevantes.")
            top_indices = [x[0] for x in top_indices]

        # 2. Busca de Força Bruta (Keyword Match) ampliada
        if not top_indices:
            logging.warning("⚠️ BM25 falhou. Tentando Busca Bruta por palavras individuais...")
            query_tokens = self.kb.index._tokenize(query)
            
            for token in query_tokens:
                mask = self.kb.df['clean_text'].str.contains(token, case=False, na=False)
                matches = self.kb.df[mask].head(2)
                if not matches.empty:
                    top_indices.extend(matches.index.tolist())
            
            # Remove duplicatas mantendo a ordem
            top_indices = list(dict.fromkeys(top_indices))[:CONFIG.MAX_RETRIEVED_DOCS]

        if not top_indices:
            logging.error("🛑 NADA encontrado nem no BM25 nem na Busca Bruta.")
            return None

        rows = self.kb.df.iloc[top_indices]
        blocks = []
        for _, row in rows.iterrows():
            block = f"\n[FONTE: {row.get('source_title', 'Vídeo do Primo')}]\n{row['clean_text']}\n"
            blocks.append(block)

        return "".join(blocks)

class DigitalBrain:
    """Gerencia a personalidade e a geração de respostas."""
    
    def __init__(self):
        if not CONFIG.API_KEY:
            st.error("⚠️ OPENROUTER_API_KEY não encontrada.")
            self.client = None
        else:
            self.client = OpenAI(base_url=CONFIG.BASE_URL, api_key=CONFIG.API_KEY)

    def _get_persona(self) -> str:
        # Prompt Engenharia de Alta Precisão para evitar respostas genéricas
        return """
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

    def think_and_speak(self, query: str, context: str):
        if not self.client: return None

        if not context:
            context = "⚠️ AVISO: Nenhuma transcrição específica foi encontrada para essa pergunta. Responda baseando-se nos princípios universais do Thiago Nigro (Buy and Hold, Diversificação, Trabalho Duro), mas avise o usuário que não há um vídeo específico sobre isso na base atual."

        full_prompt = f"CONTEXTO DE MEMÓRIA:\n{context}\n\nPERGUNTA DO SÓCIO:\n{query}"

        try:
            # Implementação da chamada OpenRouter com streaming
            return self.client.chat.completions.create(
                model=CONFIG.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_persona()},
                    {"role": "user", "content": full_prompt}
                ],
                stream=True,
                temperature=CONFIG.TEMPERATURE,
                max_tokens=4096,
                # Injeta a configuração de reasoning solicitada pelo usuário, caso o modelo suporte via API
                extra_body={
                    "reasoning": {"enabled": True} 
                }
            )
        except Exception as e:
            st.error(f"Erro neural (OpenRouter): {e}")
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
                st.caption("Gêmeo Digital (Debug Mode)")
            
            st.markdown("---")
            
            # --- ÁREA DE DIAGNÓSTICO DE DADOS (CRUCIAL) ---
            st.error("🔧 DEBUG DE DADOS")
            
            if st.session_state.kb_ref and st.session_state.kb_ref.df is not None:
                df = st.session_state.kb_ref.df
                st.write(f"📊 Total de Documentos: **{len(df)}**")
                
                # MOSTRA AS COLUNAS REAIS DO ARQUIVO
                st.write("🗂️ Colunas Encontradas:")
                st.code(list(df.columns))
                
                # TESTE DE AMOSTRA
                st.write("🔍 Amostra da 1ª linha:")
                try:
                    # Tenta mostrar a coluna clean_text se existir, senão mostra a primeira que for string
                    if 'clean_text' in df.columns:
                        st.info(df['clean_text'].iloc[0][:100] + "...")
                    else:
                        st.warning("⚠️ COLUNA 'clean_text' NÃO EXISTE!")
                        st.text(f"Use a coluna: {df.columns[0]}")
                except:
                    st.error("Erro ao ler amostra.")
            else:
                st.error("❌ DataFrame não carregado!")

            st.markdown("---")
            if st.button("🧹 Nova Conversa", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
            

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
                    
                    # --- NOVO: Expander de Debug para você ver o Contexto ---
                    with st.expander("🕵️‍♂️ O que o Primo 'lembrou'? (Contexto RAG)"):
                        if context:
                            st.text(context) # Mostra o texto exato enviado ao LLM
                        else:
                            st.warning("Nenhum contexto encontrado!")
                    # -------------------------------------------------------

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