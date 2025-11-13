# Este script carga documentos, los divide en fragmentos, 
# los convierte en embeddings y los guarda en ChromaDB.

# Este script se ejecuta una sola vez para indexar los documentos. 
# Puede probarse con un archivo .txt o .pdf de ejemplo. 

import os
from langchain.document_loaders import TextLoader, PyPDFLoader  

# TODO - Contemplar agregar otros loaders si es necesario

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import embeddings, PERSIST_DIRECTORY

def load_and_split_documents(file_path):
    """Carga un documento y lo divide en fragmentos."""
    # Detectar tipo de archivo
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_path}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def ingest_documents(file_paths):
    """Procesa una lista de archivos y los almacena en ChromaDB."""
    all_chunks = []
    for file_path in file_paths:
        print(f"Cargando archivo: {file_path}")
        chunks = load_and_split_documents(file_path)
        all_chunks.extend(chunks)

    print(f"Total de fragmentos: {len(all_chunks)}")

    # Crear o cargar ChromaDB
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Base de datos vectorial creada y persistida en:", PERSIST_DIRECTORY)

# TODO - Actualizar lista de documentos a cargar
if __name__ == "__main__":
    # Archivos de ejemplo (cambia las rutas a tus archivos de prueba)
    # Asegúrate de tener al menos un archivo de prueba en la carpeta data/
    example_files = [
        "data/ejemplo1.txt",  # Cambia por la ruta real
        # "data/ejemplo2.pdf",
    ]

    # Verificar que los archivos existen
    for f in example_files:
        if not os.path.exists(f):
            print(f"Advertencia: Archivo no encontrado: {f}. Crea este archivo o cambia la ruta.")
            # Puedes crear un archivo de ejemplo si no lo tienes
            with open(f, "w", encoding="utf-8") as temp:
                temp.write("Este es un archivo de ejemplo para probar el sistema RAG.\nContiene información educativa de prueba.")

    ingest_documents(example_files)