import streamlit as st
from vecstore import load_or_create_vectorstore
from llm import ask_model


# Configuración general
st.cache_resource.clear()
st.set_page_config(
    page_title="TP Integral - Sistema de Consulta",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Trabajo Práctico Integral – Sistema de Consulta")
st.write("Consultá cualquier cosa sobre los documentos procesados en la ingesta.")

# =====================
# 1) Cargar el Vectorstore
# =====================
@st.cache_resource
def load_vectorstore():
    st.write("📦 Cargando base vectorial...")
    return load_or_create_vectorstore()

vectorstore = load_vectorstore()

# =====================
# 2) Entrada del usuario
# =====================
st.subheader("💬 Pregunta algo")
pregunta = st.text_input("Ingresá tu consulta sobre los documentos (en español):")

# =====================
# 3) Botón de consulta
# =====================
if st.button("Consultar"):
    if not pregunta.strip():
        st.warning("Escribí una pregunta primero.")
    else:
        with st.spinner("Buscando información y generando respuesta..."):
            # Llamamos al modelo + RAG
            respuesta, retrieved_chunks = ask_model(
                pregunta=pregunta,
                vectorstore=vectorstore,
                return_sources=True
            )

        # =====================
        # Mostrar respuesta
        # =====================
        st.subheader("🧩 Respuesta del sistema")
        st.success(respuesta)

        # =====================
        # Mostrar fuentes
        # =====================
        st.subheader("📚 Fragmentos relevantes usados")
        for i, chunk in enumerate(retrieved_chunks, 1):
            with st.expander(f"Fuente #{i}"):
                st.write(chunk["page_content"])
                st.caption(f"(Documento: {chunk.get('source', 'desconocido')} | Chunk {chunk.get('chunk_id')})")
