# Esta es la aplicación Streamlit. 
# Crea la interfaz conversacional para hacer preguntas al sistema RAG.

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from config import llm, embeddings, PERSIST_DIRECTORY

# Cargar la base de datos vectorial 
# (debe haber sido creada previamente con ingest_documents.py)
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

# Cargar vectorstore y crear cadena RAG
vectorstore = load_vectorstore()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Recupera 3 fragmentos
    return_source_documents=True,
)

### TODO - Cambiar toda la interfaz para cubrir nuestro caso de proyecto !!!
# Quedaron items de ejemplo ya que no fue indicado sobre que trataba especificamente el corpus

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
        with st.spinner("Buscando información..."):
            try:
                resultado = qa_chain({"query": consulta})

                st.success("Consulta completada")
                st.subheader("Respuesta:")
                st.write(resultado["result"])

                st.subheader("Fuentes consultadas:")
                for i, doc in enumerate(resultado["source_documents"], 1):
                    with st.expander(f"Fuente {i}: {doc.metadata.get('source', 'Desconocido')}"):
                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Intenta reformular tu consulta.")
