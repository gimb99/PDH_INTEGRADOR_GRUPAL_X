"""
config.py
Archivo de configuración global del proyecto TPI.
Contiene modelos, rutas de directorios y parámetros utilizados
en ingestión, embeddings, retrieval y la aplicación Streamlit.
"""

import os

# -------------------------
# RUTAS PRINCIPALES
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Crear directorios si no existen
for d in [DATA_DIR, RAW_PDF_DIR, TEXT_DIR, CHUNK_DIR, VECTORSTORE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# -------------------------
# MODELO DE EMBEDDINGS
# -------------------------
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# -------------------------
# MODELO DE GENERACIÓN
# -------------------------
# Para local podés usar uno liviano; para deploy en HF cambiamos por uno hospedado.
GENERATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# -------------------------
# PARÁMETROS DE CHUNKING
# -------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# -------------------------
# CONFIG STREAMLIT
# -------------------------
APP_TITLE = "Sistema de Búsqueda y Preguntas - Proyecto TPI"

# -------------------------
# HUGGINGFACE API KEY
# -------------------------

# Opción recomendada: usar variable de entorno

HF_LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

