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
    MAX_RETRIEVED_DOCS: int = 30  # Aumentado para capturar mais matches
    CHUNK_SIZE: int = 300  # Tamanho do chunk em palavras (~1 minuto de fala)
    CHUNK_OVERLAP: int = 50 # Sobreposição para contexto
    SIMILARITY_THRESHOLD: float = 0.7  # Threshold de distância coseno (0-2, menor = melhor)
    
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
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_DIR)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="primo_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.video_titles = {}
        self._build_video_index()

    def _build_video_index(self):
        try:
            data = self.collection.get(include=['metadatas'])
            for meta in data.get('metadatas', []):
                vid = meta.get('video_id')
                # Tenta pegar o título de 'source' ou 'source_title'
                title = meta.get('source_title') or meta.get('source')
                if vid and title:
                    self.video_titles[vid] = title
            logging.info(f"Índice de vídeos construído: {len(self.video_titles)} vídeos encontrados.")
        except Exception as e:
            logging.error(f"Erro ao construir índice de vídeos: {e}")

    def find_videos_by_title_match(self, query: str) -> List[tuple]:
        """
        Retorna lista de (video_id, score) ordenado por relevância.
        Usa Jaccard Similarity (Interseção sobre União) simplificada.
        """
        query_words = set(query.lower().split())
        stop_words = {'a', 'o', 'e', 'do', 'da', 'de', 'em', 'para', 'que', 'os', 'as', 'um', 'uma', 'sobre', 'video', 'fala', 'resuma', 'explique'}
        query_clean = query_words - stop_words
        
        if not query_clean:
            return []
            
        matches = []
        for vid, title in self.video_titles.items():
            title_lower = title.lower()
            title_words = set(title_lower.split())
            title_clean = title_words - stop_words

            if not title_clean: continue

            # Palavras em comum entre a query e o título
            common = query_clean.intersection(title_clean)
            
            # Score 1: Quanto da query está coberta pelo título?
            coverage_query = len(common) / len(query_clean)
            
            # Score 2: Quanto do título está na query?
            coverage_title = len(common) / len(title_clean)
            
            # Score final (damos peso maior se o título inteiro estiver na query)
            final_score = max(coverage_query, coverage_title)

            # Threshold: Pelo menos 40% de match para considerar
            if final_score > 0.4:
                matches.append((vid, final_score, title))
        
        # Ordena do maior score para o menor
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    # ... (Mantenha os métodos add_texts, query, get_all_chunks_by_video, count, get_existing_source_ids IGUAIS) ...
    def add_texts(self, texts, metadatas, ids):
        # ... (seu código original aqui) ...
        if not texts: return
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.upsert(documents=texts[i:end], metadatas=metadatas[i:end], ids=ids[i:end])
    
    def query(self, query_text, n_results=5, threshold=None):
        # ... (seu código original aqui) ...
        if threshold is None: threshold = CONFIG.SIMILARITY_THRESHOLD
        results = self.collection.query(query_texts=[query_text], n_results=n_results, include=['documents', 'metadatas', 'distances'])
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            filtered = []
            for doc, meta, dist in zip(docs, metas, distances):
                if dist <= threshold: filtered.append((doc, meta, dist))
            return filtered
        return []

    def get_all_chunks_by_video(self, video_id):
        # ... (seu código original aqui) ...
        try:
            results = self.collection.get(where={"video_id": video_id}, include=['documents', 'metadatas'])
            if results and results['documents']:
                return list(zip(results['documents'], results['metadatas']))
        except: pass
        return []

    def count(self): return self.collection.count()
    
    def get_existing_source_ids(self):
        # ... (seu código original aqui) ...
        try:
            data = self.collection.get(include=['metadatas'])
            ids = data.get('ids', [])
            video_ids = set()
            for chunk_id in ids:
                if "_" in chunk_id: video_ids.add(chunk_id.rsplit('_', 1)[0])
            return list(video_ids)
        except: return []
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
            # Extração robusta do texto (adaptada para múltiplos formatos)
            # Prioridade: 'transcript' (novo formato) > 'text' (formato antigo) > 'clean_text'
            raw_text = row.get('transcript', row.get('text', row.get('clean_text', '')))
            
            # Se 'transcript' for uma lista de dicts (formato ScrapeCreators)
            if isinstance(raw_text, list):
                # Junta todos os textos dos segmentos
                raw_text = " ".join([seg.get('text', '') for seg in raw_text if isinstance(seg, dict) and 'text' in seg])
            
            # Se for JSON string, tenta parsear
            elif isinstance(raw_text, str) and (raw_text.startswith('[') or raw_text.startswith('{')):
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
                all_metas.append({
                    "source": source,
                    "url": row.get('url', ''),
                    "video_id": row.get('video_id', '')
                })

        if all_docs:
            self.memory.add_texts(all_docs, all_metas, all_ids)
            return True
        return False

    def load(self) -> bool:
        # Verifica se já temos dados indexados
        if self.memory.count() > 0:
            logging.info(f"✅ ChromaDB já carregado com {self.memory.count()} chunks.")
            return True

        # Se não, tenta carregar do Parquet/JSON
        if os.path.exists(self.config.DB_FILE):
            try:
                df = pd.read_parquet(self.config.DB_FILE)
                return self.process_and_index(df)
            except Exception as e:
                logging.error(f"Erro ao ler Parquet: {e}")
                # Continua mesmo com erro, o Auto-Sync vai popular

        # Tenta achar JSON se não tiver Parquet
        # Prioriza o arquivo de backup específico que o usuário mencionou
        specific_backup = "transcricoes_primorico_200.json"
        if os.path.exists(specific_backup):
             try:
                logging.info(f"📂 Carregando backup manual: {specific_backup}")
                df = pd.read_json(specific_backup)
                return self.process_and_index(df)
             except Exception as e:
                 logging.error(f"Erro ao ler JSON de backup: {e}")

        json_files = [f for f in os.listdir(".") if f.endswith(".json") and "transcricoes" in f]
        if json_files:
            try:
                df = pd.read_json(json_files[0])
                return self.process_and_index(df)
            except Exception as e:
                logging.error(f"Erro ao ler JSON: {e}")
                # Continua mesmo com erro

        # Se chegou aqui, a base está vazia mas o sistema pode iniciar
        # O Auto-Sync vai popular os dados posteriormente
        logging.warning("⚠️ Base de conhecimento vazia. Use o Auto-Sync para popular.")
        return True  # Retorna True para permitir inicialização

# --- 4. ENGINE DE BUSCA ---

class NeuralSearchEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def search(self, query: str) -> tuple:
        """Busca inteligente: Se achar vídeo específico, foca nele. Senão, busca geral."""
        
        # 1. Tenta identificar Intenção de Vídeo Específico
        title_matches = self.kb.memory.find_videos_by_title_match(query)
        
        hybrid_results = []
        referencias = []
        target_video_found = False

        # Se achou um vídeo com alta confiança (> 0.6), foca SÓ nele
        if title_matches and title_matches[0][1] > 0.6:
            target_vid, score, title = title_matches[0]
            logging.info(f"🎯 Vídeo Alvo Identificado: '{title}' (Score: {score:.2f})")
            
            # Pega TODOS os chunks desse vídeo
            video_chunks = self.kb.memory.get_all_chunks_by_video(target_vid)
            
            if video_chunks:
                target_video_found = True
                # Adiciona tudo ao resultado
                hybrid_results.extend(video_chunks)
                referencias.append({'titulo': title, 'url': video_chunks[0][1].get('url', '')})
        
        # 2. Se NÃO achou vídeo específico (ou o score foi baixo), faz busca semântica normal
        if not target_video_found:
            logging.info("🔍 Busca Semântica Geral iniciada...")
            semantic_results = self.kb.memory.query(query, n_results=CONFIG.MAX_RETRIEVED_DOCS)
            
            # Processa resultados semânticos
            seen_chunks = set()
            for result in semantic_results:
                doc = result[0]
                meta = result[1]
                if doc not in seen_chunks:
                    hybrid_results.append((doc, meta))
                    seen_chunks.add(doc)
                    
                    # Adiciona referência se for nova
                    ref = {'titulo': meta.get('source', 'Vídeo'), 'url': meta.get('url', '')}
                    if ref not in referencias:
                        referencias.append(ref)

        if not hybrid_results:
            return None, []

        # 3. Montagem do Contexto Final
        context_blocks = []
        
        # Se for vídeo específico, ordenamos sequencialmente (assumindo que chunks têm ordem no ID)
        if target_video_found:
             # Tenta ordenar os chunks pelo ID (ex: vid_0, vid_1, vid_2) para manter cronologia
             try:
                 hybrid_results.sort(key=lambda x: int(x[1].get('video_id', '0').split('_')[-1]) if '_' in x[1].get('video_id', '') else 0)
             except:
                 pass # Se falhar a ordenação, vai como está
        
        for text, meta in hybrid_results:
            source = meta.get('source', 'Fonte Desconhecida')
            context_blocks.append(f"FONTE: {source}\nTRECHO: {text}")
            
        # Junta tudo. DeepSeek tem contexto grande, então podemos ser generosos aqui.
        final_context = "\n\n".join(context_blocks)
        
        return final_context, referencias

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
Você É o Thiago Nigro, o Primo Rico. Não é um assistente imitando - você É ele.

## SUA IDENTIDADE
Você é o maior educador financeiro do Brasil, autor best-seller de "Do Mil ao Milhão", fundador do canal Primo Rico com milhões de inscritos. Você tem uma missão: ajudar as pessoas a conquistarem a LIBERDADE FINANCEIRA.

## SEU JEITO DE FALAR
- **Direto e sem enrolação**: Vai direto ao ponto, não fica rodeando
- **Tom de papo com amigo**: Fala como se estivesse conversando com um primo mesmo, usa "cara", "olha só", "te falo uma coisa"
- **Motivador e energético**: Usa frases de impacto, CAIXA ALTA quando quer enfatizar algo importante
- **Prático e didático**: Sempre dá exemplos concretos, usa números, situações do dia a dia
- **Usa suas frases marcantes naturalmente**:
  - "Preço é o que você paga, VALOR é o que você recebe"
  - "Quem planta valor, colhe resultado"
  - "Não existe oportunidade desperdiçada: se você não aproveitar, alguém vai"
  - "O primeiro passo pra ficar rico é gastar MENOS do que ganha"
  - "O tempo é seu maior aliado nos investimentos - os juros compostos são a oitava maravilha do mundo"
  - "Trabalhe pelo FUTURO, não pelo final de semana"

## PRINCÍPIOS QUE VOCÊ DEFENDE
- **Skin in the Game**: Só investe em algo que você também colocaria seu dinheiro
- **Antifragilidade**: Ficar mais forte com os tombos, usar as crises a seu favor
- **Juros Compostos**: O poder de deixar o dinheiro trabalhar por você
- **Diversificação inteligente**: Não colocar todos os ovos na mesma cesta
- **Educação antes de ação**: Primeiro aprende, depois investe
- **Consistência vence talento**: Fazer pouco todo mês é melhor que fazer muito uma vez

## REGRAS ABSOLUTAS
1. **RESPONDA EXCLUSIVAMENTE com base nas transcrições dos seus vídeos** - O contexto fornecido é sua ÚNICA fonte de verdade
2. **RESPONDA EXATAMENTE O QUE FOI PERGUNTADO** - Não divague, não mude de assunto
3. Se o contexto não tiver informação suficiente, seja honesto: "Cara, nos meus vídeos eu ainda não falei especificamente sobre isso, mas deixa eu te dar uma direção baseada no que eu ensino..."
4. Nunca dê conselho de investimento específico tipo "compra X ação". Você educa, não dá call
5. Responda SEMPRE em Português do Brasil
6. Seja DETALHADO e ROBUSTO nas respostas - explique o "porquê" das coisas
        """

    def think_and_speak(self, query: str, context: str):
        if not self.client: return None

        safe_query = self._sanitize_input(query)
        
        if not context:
            context = "Nenhuma informação específica encontrada na base de conhecimento sobre este tema."

        # Prompt estruturado com ênfase em relevância e exclusividade
        user_prompt = f"""## PERGUNTA DO USUÁRIO
{safe_query}

## CONTEXTO EXCLUSIVO DOS SEUS VÍDEOS (sua ÚNICA fonte de informação)
{context}

## INSTRUÇÕES OBRIGATÓRIAS
1. Use APENAS as informações do contexto acima para responder
2. RESPONDA DIRETAMENTE a pergunta - ela é sua prioridade máxima
3. Se a pergunta for sobre um tema específico, FOQUE nesse tema
4. Seja detalhado, prático e dê exemplos concretos
5. Use seu jeito característico de falar
6. NÃO invente informações que não estão no contexto

Agora responde aí, primo:"""
        
        messages = [
            {"role": "system", "content": self._get_persona()},
            {"role": "user", "content": user_prompt}
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
        # Configuração da página com o logo do Grupo Primo na aba
        st.set_page_config(
            page_title="Primo.AI 2.0 | RAG Neural",
            page_icon="Logo_primo.png",
            layout="wide"
        )
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
            # Logo
            if os.path.exists(CONFIG.LOGO_PATH):
                st.image(CONFIG.LOGO_PATH)
            else:
                st.markdown("# 🧠 Primo.AI")
            
            st.markdown("---")
            
            # Descrição curta
            st.markdown("""
            **Seu primo digital de finanças**
            
            Pergunte sobre investimentos, 
            finanças pessoais e educação financeira.
            """)
            
            st.markdown("---")
                
            # Botão limpar conversa
            if st.button("🗑️ Nova Conversa", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    def run(self):
        self.initialize_session()
        self.render_sidebar()
        self.search_engine = NeuralSearchEngine(st.session_state.kb_ref)

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
                
                # 1. Recuperação Neural - agora retorna (contexto, referencias)
                context, referencias = self.search_engine.search(prompt)
                
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
                    
                    # Adiciona referências ao final da resposta
                    if referencias:
                        refs_text = "\n\n---\n**📺 Referências:**\n"
                        for ref in referencias:
                            titulo = ref.get('titulo', 'Vídeo')
                            url = ref.get('url', '')
                            if url:
                                refs_text += f"- [{titulo}]({url})\n"
                            else:
                                refs_text += f"- {titulo}\n"
                        full_response += refs_text
                    
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    app = PrimoInterface()
    app.run()