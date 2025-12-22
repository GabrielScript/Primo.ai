# Primo.AI: O Gêmeo Digital do Primo Rico

## Descrição do Projeto

O Primo.AI é um projeto inovador que simula um "Gêmeo Digital" do Thiago Nigro (o "Primo Rico"), utilizando técnicas avançadas de Processamento de Linguagem Natural (PLN) e Geração Aumentada por Recuperação (RAG). A aplicação permite interagir com uma inteligência artificial que responde a perguntas sobre investimentos, carreira e negócios, emulando o estilo, o conhecimento e a personalidade de Thiago Nigro.

Este projeto é dividido em duas partes principais:

1.  **Coleta e Processamento de Transcrições (`Transcript_Channel.py`):** Um script robusto para extrair transcrições de vídeos do YouTube de um canal específico (neste caso, o do Primo Rico). Essas transcrições formam a base de conhecimento sobre a qual o Gêmeo Digital é treinado e consultado.
2.  **Aplicação Interativa (`app.py`):** Uma interface web construída com Streamlit que carrega as transcrições (ou outros documentos como PDF, TXT, JSON, Parquet) e permite aos usuários fazerem perguntas. A aplicação utiliza um algoritmo BM25 otimizado para recuperar informações relevantes do vasto corpus de conhecimento e, em seguida, utiliza um Large Language Model (LLM) (DeepSeek) para gerar respostas coerentes e personalizadas, mantendo a persona do Primo Rico.

O objetivo é fornecer uma ferramenta interativa para acessar o conhecimento e a perspectiva de Thiago Nigro de forma rápida e eficiente.

## Funcionalidades

### Gêmeo Digital (Primo.AI - `app.py`)
*   **Interface Conversacional:** Interaja com o Gêmeo Digital do Primo Rico através de um chat intuitivo, simulando uma conversa real.
*   **Base de Conhecimento RAG:** Utiliza um sistema de Recuperação Aumentada por Geração (RAG) para buscar informações relevantes de uma base de dados robusta antes de gerar respostas.
*   **Suporte a Múltiplos Formatos de Documentos:** Carregue conhecimentos em diversos formatos, incluindo JSON, Parquet, PDF e TXT.
*   **BM25 Otimizado:** Incorpora um algoritmo BM25 customizado com stopwords em português para uma recuperação de contexto precisa e eficiente.
*   **Persona Personalizada:** O LLM é configurado para emular a linguagem, o tom e a personalidade de Thiago Nigro, fornecendo conselhos financeiros e de negócios com seu estilo característico.
*   **Gestão de Contexto Robusta:** Implementa "freios de segurança" para gerenciar o tamanho do contexto enviado ao LLM, otimizando o uso de tokens e prevenindo sobrecarga.
*   **Histórico de Chat:** Permite visualizar e baixar o histórico completo da conversa para referência futura.
*   **Limpeza de Chat:** Opção para reiniciar a conversa a qualquer momento.
*   **Branding Visual:** Interface com elementos visuais que remetem à marca "Primo Rico".

### Coletor de Transcrições (Scraper - `Transcript_Channel.py`)
*   **Extração de Transcrições de YouTube:** Automatiza a coleta de transcrições de vídeos de qualquer canal do YouTube (mediante configuração do `CHANNEL_HANDLE`).
*   **API Robusta:** Utiliza a API `scrapecreators.com` com tratamento de retries e exponencial backoff para garantir a resiliência na coleta de dados.
*   **Parâmetros Configuráveis:** Defina o número máximo de vídeos a serem coletados e a ordem de busca (mais recentes ou mais populares).
*   **Geração de JSON:** Salva as transcrições coletadas (juntamente com metadados como título, URL, visualizações, etc.) em um arquivo JSON estruturado, pronto para ser consumido pela aplicação principal.
*   **Barra de Progresso:** Feedback visual sobre o processo de coleta através de barras de progresso (`tqdm`).

## Tecnologias Utilizadas

*   **Python 3.x**
*   **Streamlit:** Para a construção da interface web interativa (`app.py`).
*   **DeepSeek API (via OpenAI Python Client):** Como o Large Language Model (LLM) subjacente para geração de texto.
*   **Pandas:** Para manipulação e processamento de dados.
*   **PyPDF2:** Para leitura de arquivos PDF.
*   **python-dotenv:** Para gerenciamento de variáveis de ambiente.
*   **requests:** Para requisições HTTP na coleta de transcrições.
*   **urllib3 (Retry):** Para resiliência em requisições HTTP.
*   **tqdm:** Para barras de progresso visuais durante a coleta de dados.
*   **ScrapeCreators API:** Serviço externo utilizado para extrair transcrições do YouTube (`Transcript_Channel.py`).

## Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

### Pré-requisitos

*   **Python 3.x**
*   **Gerenciador de Pacotes pip**

### 1. Clonar o Repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd Primo Rico MVP
```

### 2. Criar e Ativar o Ambiente Virtual (Recomendado)

```bash
python -m venv .venv
# No Windows
.venv\Scripts\activate
# No macOS/Linux
source .venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto e adicione suas chaves de API:

```
# .env
DEEPSEEK_API_KEY="SUA_CHAVE_DE_API_DEEPSEEK"
SCRAPECREATORS_API_KEY="SUA_CHAVE_DE_API_SCRAPECREATORS"
```

*   Obtenha sua `DEEPSEEK_API_KEY` em [DeepSeek AI](https://www.deepseek.com/).
*   Obtenha sua `SCRAPECREATORS_API_KEY` em [ScrapeCreators](https://scrapecreators.com/).

### 5. Configurar o Streamlit (Opcional, para personalização)

O Streamlit permite a criação de um arquivo `secrets.toml` e `config.toml` dentro de uma pasta `.streamlit` para gerenciar segredos e configurações da aplicação.

**`.streamlit/secrets.toml`:**
```toml
# secrets.toml
DEEPSEEK_API_KEY="SUA_CHAVE_DE_API_DEEPSEEK"
SCRAPECREATORS_API_KEY="SUA_CHAVE_DE_API_SCRAPECREATORS"
```
**`.streamlit/config.toml`:**
(Este arquivo já deve existir, pode ser editado para customizações adicionais do Streamlit)
```toml
[server]
port = 8501
headless = true

[global]
enableCORS = true
enableXsrfProtection = true

[client]
toolbarMode = "minimal"

[theme]
base="dark"
primaryColor="#F63366"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"
```

**Nota:** As chaves de API podem ser carregadas tanto pelo `.env` (para scripts gerais e ambiente local) quanto pelo `secrets.toml` (para Streamlit, especialmente em deploy). Recomenda-se usar `.env` para desenvolvimento local e `secrets.toml` para deployments do Streamlit.

## Como Usar

O projeto consiste em duas partes principais: o script de coleta de transcrições e a aplicação interativa.

### 1. Coletando Transcrições (Usando `Transcript_Channel.py`)

Antes de usar a aplicação principal, você precisará de uma base de conhecimento. As transcrições do YouTube são um excelente ponto de partida.

1.  **Edite `Transcript_Channel.py`:** Abra o arquivo `Transcript_Channel.py` e configure as seguintes variáveis:
    *   `SCRAPECREATORS_API_KEY`: Certifique-se de que sua chave de API está configurada no `.env` ou diretamente no script.
    *   `CHANNEL_HANDLE`: Defina o "handle" do canal do YouTube que deseja transcrever (ex: `"primorico"`).
    *   `MAX_VIDEOS`: Especifique quantos vídeos você quer transcrever.
    *   `SORT_BY`: Escolha entre `"latest"` (mais recentes) ou `"popular"` (mais populares).

2.  **Execute o script:**
    ```bash
    python Transcript_Channel.py
    ```
    O script irá coletar as transcrições e salvará um arquivo JSON (ex: `transcricoes_primorico_17.json`) na pasta raiz do projeto. Este arquivo será sua base de conhecimento para o Primo.AI.

### 2. Iniciando a Aplicação Primo.AI (Usando `app.py`)

Após ter seu arquivo JSON de transcrições (ou outros documentos), você pode iniciar a aplicação interativa:

1.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Isso abrirá a aplicação em seu navegador padrão.

2.  **Upload de Documentos:**
    *   Na barra lateral esquerda, na seção "📂 Base de Conhecimento", clique em "Upload Arquivos".
    *   Selecione o arquivo JSON gerado pelo `Transcript_Channel.py` ou qualquer outro arquivo de texto (`.txt`), PDF (`.pdf`) ou Parquet (`.parquet`) que contenha informações relevantes. Você pode fazer upload de múltiplos arquivos.
    *   Aguarde enquanto a aplicação processa os documentos e constrói o índice de busca. Uma barra de progresso será exibida.

3.  **Interagindo com o Gêmeo Digital:**
    *   Uma vez que os documentos forem processados, você pode começar a fazer perguntas na caixa de chat na parte inferior da tela.
    *   O Primo.AI buscará informações no seu conhecimento carregado e gerará uma resposta.

4.  **Gerenciando a Conversa:**
    *   Na barra lateral, você encontrará opções para "📥 Baixar Histórico" (salva a conversa atual em um arquivo de texto) e "🗑️ Limpar Chat" (reinicia a conversa).

## Estrutura do Código

### `app.py` - Aplicação Principal do Gêmeo Digital

Este arquivo contém a lógica central da aplicação web interativa Streamlit, que permite aos usuários interagir com o Gêmeo Digital do Thiago Nigro.

**Principais Seções:**

*   **1. Configuração & Hiperparâmetros:** Define variáveis globais para a aplicação, como `LOGO_PATH`, chaves de API (carregadas via `dotenv`), o modelo de LLM (`deepseek-chat`), temperatura, e limites de segurança (`MAX_SAFE_TOKENS`, `MAX_SAFE_CHARS`) para o contexto.
*   **2. Algoritmo BM25 Otimizado:** Implementa uma versão leve do algoritmo BM25 (`SimpleBM25`) para recuperação de informações. Este algoritmo é otimizado com uma lista de stopwords em português para melhorar a relevância dos resultados da busca dentro do corpus de documentos.
*   **3. Camada de Dados & Processamento (`load_and_index_data`):** Gerencia o carregamento e a indexação dos arquivos de dados (JSON, Parquet, PDF, TXT) que formam a base de conhecimento do LLM. Realiza a normalização, limpeza de texto, identificação de colunas relevantes e criação do índice BM25 a partir do corpus consolidado. Utiliza `st.cache_data` para otimizar o desempenho.
*   **4. Busca Segura (Safety Brakes - `retrieve_context`):** Função responsável por buscar no corpus os trechos de texto mais relevantes (`MAX_RETRIEVED_DOCS`) para a pergunta do usuário, utilizando o BM25. Inclui lógica para expandir o contexto com base em uma "janela de contexto" (`CONTEXT_WINDOW_SIZE`) e garante que o contexto não exceda `MAX_SAFE_CHARS`, prevenindo estouros de token no LLM. Também filtra contextos por fonte para evitar mistura indevida de informações.
*   **5. Geração Robusta (LLM - `generate_response`):** Interage com a API do DeepSeek LLM. Define uma `system_persona` detalhada para emular o Thiago Nigro, incluindo diretrizes para o estilo de resposta. Constrói o prompt completo combinando a pergunta do usuário e o contexto recuperado, enviando-o para o LLM. Utiliza `tenacity` para retries em caso de falhas na API.
*   **6. Utilitários (Histórico - `convert_history_to_txt`):** Funções auxiliares, como a de converter o histórico do chat em um formato de texto para download.
*   **7. UI (Streamlit - `main`):** A função principal que constrói a interface do usuário Streamlit. Inclui o layout da página, cabeçalho com logo, barra lateral para upload de arquivos e gestão de conversa (limpar chat, baixar histórico), e a área de chat onde as perguntas são feitas e as respostas exibidas. Gerencia o estado da sessão (`st.session_state`) para persistir dados e mensagens.

### `Transcript_Channel.py` - Script de Coleta de Transcrições do YouTube

Este script é uma ferramenta autônoma para coletar transcrições de vídeos de um canal específico do YouTube, utilizando a API `scrapecreators.com`. Ele é crucial para a fase de preparação de dados do projeto Primo.AI.

**Principais Seções:**

*   **Configurações da Missão:** Define variáveis chave como `SCRAPECREATORS_API_KEY`, `CHANNEL_HANDLE` (o identificador do canal), `MAX_VIDEOS` (número máximo de vídeos a coletar) e `SORT_BY` (critério de ordenação: `latest` ou `popular`).
*   **Sessão de Requisições Robusta (`create_session_with_retry`):** Cria uma sessão HTTP que implementa uma política de retries com backoff exponencial. Isso garante que as chamadas à API sejam resilientes a falhas de rede ou limitações de taxa.
*   **Fase 1: Coletar IDs dos Vídeos (`fetch_youtube_video_list`):** Conecta-se à API `scrapecreators.com` para listar os vídeos de um dado canal, página por página. Ele coleta metadados básicos dos vídeos e seus IDs, respeitando o limite de `MAX_VIDEOS` e o critério `SORT_BY`.
*   **Fase 2: Buscar Transcrições (`fetch_video_transcript`):** Para cada vídeo coletado na Fase 1, este componente faz uma nova chamada à API para obter a transcrição textual completa. Ele enriquece os dados do vídeo com a transcrição e filtra quaisquer erros.
*   **Função Principal (`main`):** Orquestra o fluxo do script. Primeiro, verifica se a chave de API está configurada. Em seguida, executa as Fases 1 e 2. Ao final, imprime um relatório detalhado da missão (tempo total, custos estimados, taxa de sucesso) e salva todas as transcrições coletadas e seus metadados em um único arquivo JSON, que servirá como entrada para a aplicação `app.py`. Utiliza `tqdm` para fornecer feedback de progresso ao usuário.
