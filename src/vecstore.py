"""
vecstore.py
Módulo encargado de manejar la base vectorial del proyecto usando ChromaDB.
Incluye funciones para crear, cargar y actualizar el vectorstore,
así como para inicializar el retriever utilizado por LangChain.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL


def get_embedding_function():
    """
    Retorna el modelo de embeddings configurado en config.py.
    Sentence-transformers multilingüe para soportar español e inglés.
    """
    return HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)



def load_or_create_vectorstore():
    """
    Crea o carga una base vectorial en el directorio definido.
    Si existe, la carga; si no existe, la crea vacía.
    """
    embeddings = get_embedding_function()

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def get_retriever(k=5):
    """
    Inicializa un retriever desde la base vectorial.
    k = cantidad de documentos relevantes a recuperar.
    """
    vectorstore = load_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever
