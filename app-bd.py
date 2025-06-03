from typing import List # Importa o tipo de dado List usado para anotar funções
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Ignora mensagens de alerta do tipo UserWarning (apenas para deixar a interface limpa)
import streamlit as st # Importa a biblioteca de interface web Streamlit
from llama_index.core.llms import ChatMessage # Importa classes do LlamaIndex para usar um modelo de linguagem (LLM)
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
# Permite usar asyncio dentro do Streamlit sem conflito. Asyncio permite que múltiplas tarefas rodem ao mesmo tempo, sem travar.
import nest_asyncio
nest_asyncio.apply()
# Importa o ChromaDB, um banco de dados vetorial para armazenar e buscar embeddings
import chromadb

# Inicializa o modelo de linguagem da Ollama com o modelo Mistral
llm = Ollama(model="mistral", request_timeout=420.0)
Settings.llm = llm

# Cria o cliente do banco de dados vetorial ChromaDB com persistência (armazenamento local no diretório chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Define o nome da coleção (como uma "tabela") onde os dados serão armazenados
collection_name = "triagem_hci"

# Lista as coleções existentes no banco de dados (SELECT * FROM collections)
collections = chroma_client.list_collections()

# Verifica se a coleção já existe. Se sim, obtém ela. Caso contrário, cria uma nova.
if collection_name in [col.name for col in collections]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

# Mostra o título da interface da aplicação no navegador
st.title("Assistente de Pós-Graduação UNIJUÍ")

# Cria um campo de texto onde enfermeiro(a) (ou outro profissional de saúde) pode informar os sintomas do paciente
new_case = st.text_area("Descreva seus interesses para uma Pós-Graduação e lhe retornamos um Orientador")

# Quando o botão é clicado, o sistema começa a análise
if st.button("Mostrar Resultado"):
    # Verifica se o campo de texto com os sintomas foi preenchido
    if new_case:
        # Mostra um spinner (indicador visual) enquanto o processamento ocorre
        with st.spinner("Classificando..."):

            # Importa e carrega o modelo de embeddings (vetorização) 
            from sentence_transformers import SentenceTransformer
            # Carrega um modelo pré-treinado de embeddings semânticos chamado all-MiniLM-L6-v2, 
            # disponibilizado pela Sentence Transformers, uma biblioteca baseada no Hugging Face e no PyTorch
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Função que converte um texto em vetor numérico (embedding)
            def embed_text(text: str) -> List[float]:
                # Aplica o modelo para transformar o texto em vetor (tensor)
                embeddings = model.encode([text], convert_to_tensor=True)
                # Converte o tensor para lista de floats e retorna
                return embeddings.cpu().numpy()[0].tolist()

            # Função para ler os casos de triagem simulados a partir do arquivo "casos.txt"
            def load_triagem_cases(filepath: str) -> List[str]:
                # Abre o arquivo e retorna apenas linhas não vazias
                with open(filepath, "r", encoding="utf-8") as file:
                    return [line.strip() for line in file if line.strip()]

            # Carrega os casos simulados do arquivo
            triagem_cases = load_triagem_cases("casos.txt")

            # Recupera os IDs já existentes no banco vetorial para evitar duplicação
            existing_ids = set(collection.get()["ids"])

            # Para cada caso do arquivo
            for i, case in enumerate(triagem_cases):
                # Cria um ID único com base no índice
                case_id = f"case_{i}"
                # Adiciona o caso no banco apenas se ainda não existir
                if case_id not in existing_ids:
                    # Converte o caso em embedding
                    embedding = embed_text(case)
                    # Adiciona o vetor ao banco com o ID e metadado de conteúdo textual
                    collection.add(
                        embeddings=[embedding],
                        ids=[case_id],
                        metadatas=[{"content": case}]  # Armazena o texto original como metadado
                    )

            # Converte os sintomas informados pelo usuário em vetor (embedding)
            query_embedding = embed_text(new_case)

            # Consulta no banco vetorial os 3 casos mais semelhantes ao novo caso informado.
            # O ChromaDB utiliza o vetor de embedding gerado para o novo caso (query_embedding)
            # e compara esse vetor com todos os vetores previamente armazenados na coleção.
            # Essa comparação é feita usando uma métrica de similaridade (como produto interno ou cosseno),
            # retornando os 'n_results' casos com maior similaridade semântica.
            # O resultado inclui os metadados dos casos mais parecidos, que serão usados para orientar a resposta do LLM.
            results = collection.query(query_embeddings=[query_embedding], n_results=3)

            # Extrai os conteúdos (textos) dos casos similares retornados
            similar_cases = [metadata["content"] for metadata in results['metadatas'][0]]

            # Monta o prompt com os sintomas e os casos similares
            input_text = f"Sintomas do novo caso: {new_case}\n\nCasos Similares: {' '.join(similar_cases)}"

            # Cria a sequência de mensagens para enviar ao modelo de linguagem
            messages = [
                # Mensagem inicial: define o comportamento do assistente como um profissional da saúde
                ChatMessage(
                    role="system",
                    content="Você é um Assistente para auxiliar estudantes a escolher um orientador para sua Pós-Graduação com base em seus interesses. Os professores que podem ser orientadores são funcionários da UNIJUÍ"
                ),
                # Mensagem com os sintomas e os casos similares
                ChatMessage(role="user", content=input_text),
                # Solicita resposta estruturada com classificação, justificativa e conduta
                ChatMessage(
                    role="user",
                    content="Com base nas áreas de estudo dos professores, forneça o nome do professor mais qualificado para satisfazer os interesses do aluno, justifique sua escolha com base na compatibilidade de interesses do aluno e com as áreas de pesquisa dos professores. Não inclua informações irrelevantes como ano de graduação ou locais onde o professor já estudou."
                ),
            ]

            # Tenta executar a consulta ao modelo (via Ollama)
            try:
                resposta = llm.chat(messages)  # Envia as mensagens para o modelo e recebe resposta
                st.subheader("Resultado da Análise de Compatibilidade:")
                st.write(str(resposta))  # Exibe o resultado na interface web
            except Exception as e:
                # Em caso de erro, mostra uma mensagem de erro na interface
                st.error(f"Ocorreu um erro ao consultar o modelo: {e}")
    else:
        # Caso o usuário não preencha os sintomas, exibe aviso
        st.warning("Por favor, insira suas áreas de interesse.")

