"""
ingest.py
Pipeline completo de ingesta de documentos para el Proyecto TPI.

Flujo:
1. Leer PDFs desde data/raw_pdfs/
2. Extraer texto
3. Detectar idioma
4. Traducir a español si está en inglés (Helsinki-NLP/opus-mt-en-es vía HF API)
5. Guardar texto traducido en data/texts/
6. Realizar chunking
7. Calcular embeddings con sentence-transformers
8. Guardar vectorstore en ChromaDB
"""

import os
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langdetect import detect
from src.config import HF_API_KEY

# Configuración global
from src.config import (
    RAW_PDF_DIR,
    TEXT_DIR,
    CHUNK_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from src.vecstore import load_or_create_vectorstore


# ---------------------------------------------------------
# 1. TRADUCCIÓN VIA HUGGINGFACE API (OPUS-MT EN→ES)
# ---------------------------------------------------------

def translate_text(text: str, hf_api_key: str):
    """
    Traduce texto de inglés a español usando Helsinki-NLP/opus-mt-en-es
    a través de HuggingFace Inference API.
    """

    url = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-es"

    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }

    payload = {"inputs": text}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("Error en traducción:", response.text)
        return text  # fallback: devolver texto original

    try:
        translated = response.json()[0]["translation_text"]
        return translated
    except Exception:
        return text


# ---------------------------------------------------------
# 2. PROCESAMIENTO DE PDF → TEXTO
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae todo el texto de un PDF usando PyPDFLoader.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text = ""
    for page in pages:
        text += page.page_content + "\n"

    return text


# ---------------------------------------------------------
# 3. DETECCIÓN DE IDIOMA
# ---------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Retorna 'en' para inglés, 'es' para español, etc.
    """
    try:
        return detect(text)
    except:
        return "unknown"


# ---------------------------------------------------------
# 4. CHUNKING
# ---------------------------------------------------------

def chunk_text(text: str, filename: str):
    """
    Divide el texto en chunks y los guarda en data/chunks/{filename}.json
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(text)

    outpath = os.path.join(CHUNK_DIR, f"{filename}.json")

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks


# ---------------------------------------------------------
# 5. EMBEDDINGS + CHROMA
# ---------------------------------------------------------

def embed_chunks(chunks: list, metadata: dict):
    """
    Inserta chunks embebidos en ChromaDB.
    Cada chunk es un documento independiente.
    """
    vectorstore = load_or_create_vectorstore()

    docs = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        docs.append(chunk)
        metadatas.append({**metadata, "chunk_id": i})

    vectorstore.add_texts(docs, metadatas=metadatas)

    # guardar cambios
    vectorstore.persist()


# ---------------------------------------------------------
# 6. PIPELINE COMPLETO DE INGESTA
# ---------------------------------------------------------

def ingest_documents(hf_api_key: str):
    """
    Procesa todos los PDFs en data/raw_pdfs/.
    Traduce, chunkifica y embebe.
    """

    pdf_files = [
        f for f in os.listdir(RAW_PDF_DIR) 
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No hay PDFs en data/raw_pdfs/")
        return

    print(f"[INFO] Procesando {len(pdf_files)} PDFs...\n")

    for pdf in pdf_files:
        print(f"=== Procesando {pdf} ===")

        pdf_path = os.path.join(RAW_PDF_DIR, pdf)
        filename = pdf.replace(".pdf", "")

        # Extraer texto
        text = extract_text_from_pdf(pdf_path)

        # Detectar idioma
        lang = detect_language(text)
        print(f" → Idioma detectado: {lang}")

        # Traducir si es inglés
        if lang == "en":
            print(" → Traduciendo a español...")
            text = translate_text(text, hf_api_key)
        else:
            print(" → No requiere traducción.")

        # Guardar texto traducido
        out_text_path = os.path.join(TEXT_DIR, f"{filename}.txt")
        with open(out_text_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(" → Texto guardado.")

        # Chunking
        chunks = chunk_text(text, filename)
        print(f" → {len(chunks)} chunks generados.")

        # Embeddings + Vectorstore
        embed_chunks(chunks, metadata={"source": filename})
        print(" → Embeddings generados y guardados en ChromaDB.\n")

    print("\n[OK] Ingesta completada.\n")


# ---------------------------------------------------------
# 7. EJECUCIÓN DIRECTA
# ---------------------------------------------------------

if __name__ == "__main__":
    if not HF_API_KEY:
        raise ValueError(
            "No se encontró HF_API_KEY. "
            "Configuralo en config.py o como variable de entorno."
        )

    ingest_documents(HF_API_KEY)