# Assistente de IA para Escolher um Orientador de Pós-Graduação

Este projeto implementa um assistente de IA com interface web para scolher um Orientador de Pós-Graduação automatizado, utilizando **embeddings semânticos com BioBERTpt(clin)**, **banco vetorial (ChromaDB)** e um **modelo de linguagem local (Mistral via Ollama)**.


---

## Visão Geral

O assistente recebe **áreas de interesse do aluno** como entrada, busca **áreas de estudo/atuação semelhantes dos professores** em um banco vetorial e retorna um professor que seja mais adequado para ser orientador.

## Pré-requisitos

- **Sistema operacional:** Windows 10 ou superior  
- **Memória RAM recomendada:** 8 GB ou mais  
- **Espaço em disco:** 10 GB ou mais  

---

## Etapas de Instalação

### 1. Instalar o Ollama

> O Ollama roda modelos LLM localmente. Requer suporte à virtualização.

- Baixe: [https://ollama.com/download](https://ollama.com/download)
- Após a instalação, teste no terminal:

```bash
ollama list
```

---

### 2. Baixar o modelo `mistral`

```bash
ollama pull mistral
```

> Obs.: O modelo `llama3` pode exigir mais memória RAM.

---

### 3. Instalar Python 3.10.x

- Baixe: [Python 3.10.11](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
- Marque a opção **Add Python to PATH**
- Verifique a versão instalada:

```bash
python --version
```

---

### 4. Criar ambiente virtual

```powershell
cd "C:\Users\SEU_USUARIO\Documents\Projetos\AgenteIA"
python -m venv venv310
.\venv310\Scripts\Activate.ps1
```

---

### 5. Instalar dependências

```bash
pip install --upgrade pip
pip install streamlit
pip install llama-index
pip install chromadb
pip install sentence-transformers
pip install nest_asyncio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python.exe -m pip install --upgrade pip
pip install --upgrade llama-index
pip install llama-index-llms-ollama
```

---

## Executar o Aplicativo

```powershell
streamlit run app-bd-clin.py
```

Abra o navegador em:

[http://localhost:8501](http://localhost:8501)

## Visualizar dados com DB Browser for SQLite

1. Baixe: [https://sqlitebrowser.org](https://sqlitebrowser.org)
2. Abra o diretório `./chroma_db/`
3. Execute consultas SQL como:

```sql
SELECT * FROM collections;
SELECT * FROM embeddings;
SELECT * FROM embeddings_queue;
```
## Links Úteis

- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)
- [ChromaDB](https://docs.trychroma.com/)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [BioBERTpt(clin)](https://aclanthology.org/2020.clinicalnlp-1.7.pdf)
