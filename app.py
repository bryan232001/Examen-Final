import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from huggingface_hub import snapshot_download
from google import genai

# --------------------------------------------------------------
# Configuración de página
# --------------------------------------------------------------
st.set_page_config(page_title="RAG arXiv Chat", page_icon="📚", layout="wide")
st.title("📚 Chat RAG sobre arXiv Paper Abstracts")
st.caption("Sistema de Recuperación de Información con embeddings + Gemini")

# --------------------------------------------------------------
# API Key desde Streamlit Secrets (NUNCA hardcodeada)
# En Streamlit Cloud: Settings -> Secrets -> GOOGLE_API_KEY = "tu_key"
# --------------------------------------------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ No se encontró GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
LLM_MODEL = "gemini-2.5-flash"

RAG_PROMPT_TEMPLATE = """Eres un asistente experto en literatura científica. Tu tarea es responder la consulta del usuario ÚNICAMENTE utilizando la información contenida en los documentos de contexto proporcionados a continuación.

Reglas:
1. Si el contexto contiene información suficiente, responde de forma clara y completa, integrando información de varios documentos si es necesario.
2. Cuando uses información de un documento, referencia su número entre corchetes, por ejemplo [Doc 1], [Doc 2].
3. Si el contexto NO contiene información suficiente para responder la consulta, indícalo explícitamente diciendo: "El corpus no contiene información suficiente para responder esta consulta con certeza." No inventes información que no esté en el contexto.

--- CONTEXTO ---
{context}
--- FIN DEL CONTEXTO ---

Consulta del usuario: {query}

Respuesta:"""


# --------------------------------------------------------------
# Carga de embeddings + índice FAISS (una sola vez, cacheado)
# --------------------------------------------------------------
class BGEEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        instruccion = "Represent this sentence for searching relevant passages: "
        return self.model.encode(instruccion + text, normalize_embeddings=True).tolist()


@st.cache_resource(show_spinner="Descargando índice FAISS desde Hugging Face...")
def load_vector_store():
    local_path = snapshot_download(
        repo_id="Bryan23y/rag-arxiv-faiss",
        repo_type="dataset",
    )
    embedding_function = BGEEmbeddings()
    vector_store = FAISS.load_local(
        local_path,
        embedding_function,
        allow_dangerous_deserialization=True,
    )
    return vector_store

vector_store = load_vector_store()


# --------------------------------------------------------------
# Funciones del pipeline RAG
# --------------------------------------------------------------
def retrieve_documents(query, k=5):
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    return docs_with_scores  # retorna (doc, score) para exponer la evidencia completa


def build_context(retrieved_documents):
    context_parts = []
    for i, doc in enumerate(retrieved_documents):
        titulo = doc.metadata.get("title", "N/A")
        categorias = doc.metadata.get("categories", "N/A")
        abstract = doc.page_content
        context_parts.append(
            f"[Doc {i+1}] Título: {titulo}\nCategorías: {categorias}\nAbstract: {abstract}"
        )
    return "\n\n".join(context_parts)


def generate_answer(query, context):
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "system_instruction": "Eres un asistente de investigación riguroso y honesto.",
        },
    )
    return response.text


def rag_query(query, k=5):
    docs_with_scores = retrieve_documents(query, k=k)
    retrieved_documents = [doc for doc, _ in docs_with_scores]
    context = build_context(retrieved_documents)
    answer = generate_answer(query, context)
    
    evidencias = [
        {
            "titulo": doc.metadata.get("title", "N/A"),
            "categorias": doc.metadata.get("categories", "N/A"),
            "fragmento": doc.page_content[:300] + "...",
            # Conversión correcta: distancia L2 de FAISS -> Similitud Coseno
            "score": round(1 - (float(score) / 2), 4),
        }
        for doc, score in docs_with_scores
    ]
    return {"query": query, "respuesta": answer, "evidencias": evidencias}


# --------------------------------------------------------------
# Estado del chat (solo historial visual, sin memoria conversacional real)
# --------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Slider para top-k en la barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    k = st.slider("Número de documentos a recuperar (k)", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "**Sobre este sistema:**\n\n"
        "Corpus: arXiv Paper Abstracts (Kaggle)\n\n"
        "Embeddings: BAAI/bge-small-en-v1.5\n\n"
        "Vector store: FAISS\n\n"
        "LLM: Gemini 2.5 Flash"
    )

# Renderizar historial de la sesión
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "evidencias" in msg:
            with st.expander("📄 Ver evidencias utilizadas"):
                for i, ev in enumerate(msg["evidencias"]):
                    st.markdown(f"**[Doc {i+1}] {ev.get('titulo', 'N/A')}**")
                    st.caption(f"Categorías: {ev.get('categorias', 'N/A')} | Similitud coseno: {ev.get('score', 'N/A')}")
                    st.write(ev.get("fragmento", ""))
                    st.markdown("---")

# --------------------------------------------------------------
# Input de chat
# --------------------------------------------------------------
if query := st.chat_input("Escribe tu consulta sobre artículos científicos..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando documentos relevantes y generando respuesta..."):
            resultado = rag_query(query, k=k)
        st.markdown(resultado["respuesta"])
        with st.expander("📄 Ver evidencias utilizadas"):
            for i, ev in enumerate(resultado["evidencias"]):
                st.markdown(f"**[Doc {i+1}] {ev.get('titulo', 'N/A')}**")
                st.caption(f"Categorías: {ev.get('categorias', 'N/A')} | Similitud coseno: {ev.get('score', 'N/A')}")
                st.write(ev.get("fragmento", ""))
                st.markdown("---")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resultado["respuesta"],
            "evidencias": resultado["evidencias"],
        }
    )
