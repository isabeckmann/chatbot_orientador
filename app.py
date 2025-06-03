from typing import List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import nest_asyncio
import chromadb

# Aplicar nest_asyncio
nest_asyncio.apply()

# Configuração do LLM
llm = Ollama(model="mistral", request_timeout=420.0)
Settings.llm = llm

# Inicializar cliente ChromaDB
chroma_client = chromadb.Client()
collection_name = "triagem_hci"
collections = chroma_client.list_collections()

# Criar ou obter coleção
if collection_name in [col.name for col in collections]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

# Interface Streamlit
st.title("Assistente de Pós-Graduação UNIJUÍ")

new_case = st.text_area("Descreva seus interesses para uma Pós-Graduação e lhe retornamos um Orientador")

if st.button("Mostrar Resultado"):
    if new_case:
        with st.spinner("Classificando..."):

            # Carrega apenas agora
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Função de embedding
            def embed_text(text: str) -> List[float]:
                embeddings = model.encode([text], convert_to_tensor=True)
                return embeddings.cpu().numpy()[0].tolist()

            # Carrega casos simulados e popula coleção
            def load_triagem_cases(filepath: str) -> List[str]:
                with open(filepath, "r", encoding="utf-8") as file:
                    return [line.strip() for line in file if line.strip()]

            triagem_cases = load_triagem_cases("casos.txt")

            for i, case in enumerate(triagem_cases):
                embedding = embed_text(case)
                collection.add(embeddings=[embedding], ids=[f"case_{i}"], metadatas=[{"content": case}])

            # Buscar casos similares
            query_embedding = embed_text(new_case)
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            similar_cases = [metadata["content"] for metadata in results['metadatas'][0]]

            # Construção do prompt
            input_text = f"Áreas de interesse do aluno: {new_case}\n\nÁreas similares dos professores: {' '.join(similar_cases)}"
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

            try:
                resposta = llm.chat(messages)
                st.subheader("Orientador sugerido:")
                st.write(str(resposta))
            except Exception as e:
                st.error(f"Ocorreu um erro ao consultar o modelo: {e}")
    else:
        st.warning("Por favor, insira suas áreas de interesse.")
