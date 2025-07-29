from typing import List
import warnings
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import nest_asyncio
import asyncio
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

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
collection_name = "orientadores"

try:
    collection = chroma_client.get_collection(collection_name)
except chromadb.errors.InternalError:
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Coleção '{collection_name}' criada.")

st.title("Assistente de Pós-Graduação UNIJUÍ")

def load_professors(filepath: str) -> List[dict]:
    try:
        professors = []
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Professor:"):
                    parts = line.split("Área de estudo:")
                    nome = parts[0].replace("Professor:", "").strip().rstrip(".")

                    area_estudo = ""
                    area_atuacao = "Não informada"

                    if len(parts) > 1:
                        sub_parts = parts[1].split("Área de atuação:")
                        area_estudo = sub_parts[0].strip().rstrip(".")
                        if len(sub_parts) > 1:
                            area_atuacao = sub_parts[1].strip().rstrip(".")

                    professor_data = {
                        "nome": nome,
                        "area_estudo": area_estudo,
                        "area_atuacao": area_atuacao
                    }
                    professors.append(professor_data)

        return professors

    except FileNotFoundError:
        st.error(f"Arquivo '{filepath}' não encontrado.")
        return []

professores = load_professors("casos.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
model = AutoModel.from_pretrained("pucpr/biobertpt-clin").to(device)

def embed_text(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().tolist()

existing_ids = set(collection.get()["ids"])

for i, professor in enumerate(professores):
    professor_id = f"professor_{i}"
    if professor_id not in existing_ids:
        try:
            if "area_atuacao" in professor:
                embedding = embed_text(professor["area_atuacao"])
                collection.add(
                    embeddings=[embedding],
                    ids=[professor_id],
                    metadatas=[{
                        "nome": professor["nome"],
                        "area_estudo": professor["area_estudo"],
                        "area_atuacao": professor["area_atuacao"]
                    }]
                )
            else:
                st.warning(f"Professor {professor['nome']} não possui área de atuação definida.")
        except RuntimeError as e:
            st.warning(f"Erro ao processar o professor {professor_id}: {e}")

# Cria um campo de texto onde o aluno pode informar suas áreas de interesse
new_case = st.text_area("Descreva suas áreas de interesse para uma Pós-Graduação e lhe retornamos um Orientador (isso pode demorar um pouquinho)")

if st.button("Mostrar Resultado"):
    if new_case:
        with st.spinner("Classificando..."):
            try:
                query_embedding = embed_text(new_case)

                results = collection.query(query_embeddings=[query_embedding], n_results=3)

                professores_similares = [
                    {"nome": metadata["nome"], "area_atuacao": metadata["area_atuacao"]} 
                    for metadata in results['metadatas'][0]
                ]

                input_text = f"Áreas de interesse do aluno: {new_case}\n\nÁreas de pesquisa dos professores: "
                input_text += "\n".join([f"{prof['nome']}: {prof['area_atuacao']}" for prof in professores_similares])

                messages = [
                    ChatMessage(
                        role="system",
                        content="Você é um assistente de inteligência artificial que ajuda estudantes de pós-graduação a escolher o orientador mais compatível com seus interesses acadêmicos. Com base nas informações fornecidas, identifique o professor cuja área de atuação melhor corresponde aos interesses do aluno. Não cite professores cujas áreas não tenham relação clara com os interesses do aluno. Evite rodeios e seja direto na recomendação."
                    ),
                    ChatMessage(role="user", content=input_text),
                    ChatMessage(
                        role="user",
                        content="Com base nas áreas de interesse do aluno e nas áreas de pesquisa dos professores, forneça o nome do professor mais qualificado para orientá-lo, justificando sua escolha com base na compatibilidade dos interesses."
                    ),
                ]

                resposta = llm.chat(messages)
                st.subheader("Orientador sugerido:")
                st.write(str(resposta))

            except Exception as e:
                st.error(f"Erro ao consultar o modelo: {e}")
    else:
        st.warning("Por favor, insira suas áreas de interesse.")
