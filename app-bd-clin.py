from typing import List # Importa o tipo de dado List usado para anotar funções
import warnings # Importa a biblioteca warnings para gerenciar avisos
warnings.filterwarnings("ignore", category=UserWarning) # Ignora mensagens de alerta do tipo UserWarning (apenas para deixar a interface limpa)
import streamlit as st # Importa a biblioteca de interface web Streamlit
from llama_index.core.llms import ChatMessage # Importa classes do LlamaIndex para usar um modelo de linguagem (LLM)
from llama_index.llms.ollama import Ollama # Importa o modelo Ollama
from llama_index.core import Settings # Importa as configurações do LLM
import nest_asyncio # Permite usar asyncio dentro do Streamlit sem conflito
import asyncio
import chromadb # Importa o ChromaDB, um banco de dados vetorial para armazenar e buscar embeddings
from transformers import AutoTokenizer, AutoModel # Importa o modelo BioBERTpt(clin) para geração de embeddings
import torch # Importa PyTorch para operações tensoriais

# Ajuste para o loop assíncrono no Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    nest_asyncio.apply()

# Inicializa o modelo de linguagem da Ollama com o modelo Mistral
llm = Ollama(model="mistral", request_timeout=420.0)
Settings.llm = llm

# Cria o cliente do banco de dados vetorial ChromaDB com persistência
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "triagem_hci"

# Verifica se a coleção já existe, caso contrário, cria uma nova
collections = chroma_client.list_collections()
if collection_name in [col.name for col in collections]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Coleção '{collection_name}' criada.")

# Mostra o título da interface da aplicação no navegador
st.title("Assistente de Pós-Graduação UNIJUÍ")

# Cria um campo de texto onde enfermeiro(a) (ou outro profissional de saúde) pode informar os sintomas do paciente
new_case = st.text_area("Descreva suas áreas de interesse para uma Pós-Graduação e lhe retornamos um Orientador (isso pode demorar um pouquinho)")

# Define o dispositivo como CUDA (GPU) se disponível, caso contrário, utiliza CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o tokenizer e o modelo sem mover para o dispositivo imediatamente
tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
model = AutoModel.from_pretrained("pucpr/biobertpt-clin")

# Move o modelo para o dispositivo
model = model.to(device)

# Função para converter texto em embedding
def embed_text(text: str) -> List[float]:
    # Processa o texto e gera embeddings
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().tolist()

# Função para ler os casos de triagem simulados a partir do arquivo "casos.txt"
def load_triagem_cases(filepath: str) -> List[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        st.error("Arquivo 'casos.txt' não encontrado.")
        return []

# Carrega os casos simulados do arquivo
triagem_cases = load_triagem_cases("casos.txt")
existing_ids = set(collection.get()["ids"])

# Insere novos casos no banco vetorial
for i, case in enumerate(triagem_cases):
    case_id = f"case_{i}"
    if case_id not in existing_ids:
        try:
            embedding = embed_text(case)
            collection.add(
                embeddings=[embedding],
                ids=[case_id],
                metadatas=[{"content": case}]
            )
        except RuntimeError as e:
            st.warning(f"Erro ao processar o caso {case_id}: {e}")

# Quando o botão é clicado, o sistema começa a análise
if st.button("Mostrar Resultado"):
    if new_case:
        with st.spinner("Classificando..."):
            try:
                # Converte os sintomas informados pelo usuário em vetor (embedding)
                query_embedding = embed_text(new_case)

                # Consulta no banco vetorial os 3 casos mais semelhantes ao novo caso informado
                results = collection.query(query_embeddings=[query_embedding], n_results=3)

                # Extrai os conteúdos (textos) dos casos similares retornados
                similar_cases = [metadata["content"] for metadata in results['metadatas'][0]]

                # Monta o prompt com os sintomas e os casos similares
                input_text = f"Áreas de interesse do aluno: {new_case}\n\nÁreas similares dos professores: {' '.join(similar_cases)}"

                # Cria a sequência de mensagens para enviar ao modelo de linguagem
                messages = [
                    ChatMessage(
                        role="system",
                        content="Você é um Assistente para auxiliar estudantes a escolher um orientador para sua Pós-Graduação com base em seus interesses. Os professores que podem ser orientadores são funcionários da UNIJUÍ."
                    ),
                    ChatMessage(role="user", content=input_text),
                    ChatMessage(
                        role="user",
                        content="Com base nas áreas de estudo dos professores, forneça o nome do professor mais qualificado para satisfazer o papel de orientador do aluno com base nos interesses do aluno, justifique sua escolha com base na compatibilidade de interesses do aluno e com as áreas de pesquisa e de atuação dos professores. Não inclua informações irrelevantes como ano de graduação ou locais onde o professor já estudou."
                    ),
                ]

                # Tenta executar a consulta ao modelo (via Ollama)
                resposta = llm.chat(messages)
                st.subheader("Orientador sugerido:")
                st.write(str(resposta))

            except Exception as e:
                st.error(f"Erro ao consultar o modelo: {e}")
    else:
        st.warning("Por favor, insira suas áreas de interesse.")
