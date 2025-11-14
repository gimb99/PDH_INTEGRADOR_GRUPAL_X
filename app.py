# Esta es la aplicación Streamlit. 
# Crea la interfaz conversacional para hacer preguntas al sistema RAG.

## Ejecutame como "streamlit run app.py" sin comillas desde el ambiente :)

import streamlit as st

# --- Imports actualizados para LangChain v0.1.0+ ---
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# ----------------------------------------------------

# Cargar embeddings (mismo modelo que usaste en ti_ctg.py)
# Asegúrate de usar el import correcto aquí también si es necesario
# from langchain.embeddings import HuggingFaceEmbeddings # <-- Anterior
# from langchain_huggingface import HuggingFaceEmbeddings # <-- Nuevo (ya está arriba)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Directorio donde se guardó la base de datos vectorial
PERSIST_DIRECTORY = "./db_vectores"

# Cargar la base de datos vectorial (debe haber sido creada previamente con ti_ctg.py)
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

# Cargar vectorstore y crear retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Recupera 3 fragmentos

# Cargar el LLM (Flan-T5 Base via Inference API)
# from langchain.llms import HuggingFaceEndpoint # <-- Anterior
# from langchain_huggingface import HuggingFaceEndpoint # <-- Ya está arriba
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",  # T5 usa text2text
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
    # token="TU_TOKEN_AQUI",  # Omitir si ya hiciste hf auth login
)

# --- RAG Chain ---
# Definir el prompt template para Augmentation
template = """
Eres un asistente experto en el dominio del corpus proporcionado. \
Responde la pregunta del usuario basándote ÚNICAMENTE en la información del contexto. \
Si la información no es suficiente, responde claramente que no encontraste la información en el corpus.

Pregunta: {question}
Contexto: {context}

Respuesta:
"""
prompt = ChatPromptTemplate.from_template(template)

# Definir la cadena RAG: Input -> Retriever -> Prompt + Context -> LLM -> Output
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# --- Fin RAG Chain ---

# Interfaz Streamlit
st.set_page_config(page_title="Sistema RAG Educativo", page_icon="📚")
st.title("📚 Sistema RAG Educativo")
st.markdown("Pregunta sobre el material didáctico cargado.")

# Ejemplos de consultas en la barra lateral
with st.sidebar:
    st.header("Ejemplos de Consultas")
    st.write("- ¿Cuál es el tema principal del documento 1?")
    st.write("- Resume el contenido del material.")
    st.write("- ¿Qué dice sobre el concepto X?")

# Input de la consulta
consulta = st.text_input("Haz tu pregunta:", placeholder="Ej: ¿De qué trata este material?")

if st.button("Consultar", type="primary"):
    if not consulta:
        st.warning("Por favor, escribe una pregunta.")
    else:
        with st.spinner("Buscando información y generando respuesta..."):
            try:
                # --- Ejecutar la cadena RAG ---
                # Recuperar documentos relevantes
                relevant_docs = retriever.invoke(consulta)
                # Generar respuesta usando la cadena
                respuesta_generada = rag_chain.invoke(consulta)
                # --- Fin ejecución ---

                st.success("Consulta completada")
                st.subheader("Respuesta:")
                st.write(respuesta_generada)

                st.subheader("Fuentes consultadas:")
                for i, doc in enumerate(relevant_docs, 1):
                    with st.expander(f"Fuente {i}: {doc.metadata.get('nombre_documento', 'Desconocido')}"):
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Intenta reformular tu consulta.")
