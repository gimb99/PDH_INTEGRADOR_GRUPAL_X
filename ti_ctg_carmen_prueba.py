# -*- coding: utf-8 -*-

## 📥 Paso 1 – Ingesta de documentos PDF y traducción automática
"""
Este paso carga los documentos técnicos desde el disco local.
Se leen PDFs tanto en español como en inglés.  
Los documentos en inglés se traducen automáticamente al español con un modelo de Hugging Face.
El resultado final es un corpus unificado en español que se utilizará en los siguientes pasos del sistema RAG.
"""

# 📌 Ingesta y traducción automática de documentos PDF

# ========================
# ✅ Librerías necesarias
# ========================
"""Inicialmente se trabajó desde Google Colab, se agregaron diversas librerías que, al pasar a 
Visual dejaron de funcionar, por lo que se cambiaron las librerías"""

from langchain_community.document_loaders import PyMuPDFLoader  # Para cargar PDFs
from transformers import pipeline   # Para traducir texto
import os   # Para acceder a archivos en disco
from langchain_community.vectorstores import Chroma



# ========================
# ✅ Modelo de traducción
# ========================
# Este modelo traduce texto de inglés a español usando Hugging Face
# Se hizo necesario agregar esta traducción porque hay documentación en inglés y en español y queremos estandarizar

from transformers import MarianMTModel, MarianTokenizer
import torch
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")


# ======================================
# 📁 Definimos las carpetas del corpus 
# ======================================
carpetas_es = "./data/corpus_base"
carpeta_en = "./data/complementos_tecnicos"

# ======================================
# 📚 Función para cargar documentos PDF
# ======================================

def cargar_documentos_desde_carpeta(rutas_carpeta):
    documentos = []
    for ruta in rutas_carpeta:
        for archivo in os.listdir(ruta):
            if archivo.endswith(".pdf"):
                ruta_completa = os.path.join(ruta, archivo)
                loader = PyMuPDFLoader(ruta_completa)
                documentos_pdf = loader.load()
                documentos.extend(documentos_pdf)
    return documentos

# ======================================
# 🌍 Función para traducir texto de inglés a español
# ======================================
# Modelo para traducir de inglés a español
modelo_trad = "Helsinki-NLP/opus-mt-en-es"
tokenizer_trad = MarianTokenizer.from_pretrained(modelo_trad)
model_trad = MarianMTModel.from_pretrained(modelo_trad)
#Cambio para hacer la traducción de otra forma, la actual me corta las palabras
"""
def traducir_texto(texto, max_length=512):
    oraciones = [texto[i:i+max_length] for i in range(0, len(texto), max_length)]
    resultado = []
    for segmento in oraciones:
        inputs = tokenizer_trad(segmento, return_tensors="pt", truncation=True)
        translated = model_trad.generate(**inputs)
        texto_traducido = tokenizer_trad.decode(translated[0], skip_special_tokens=True)
        resultado.append(texto_traducido)
    return " ".join(resultado)
    """ 
def traducir_texto(texto, max_tokens=None, stride=50):#deja un solapamiento de 50 tokens entre segmentos para que no se corte una frase justo al final. Si notás que aparecen cortes, aumentalo; si el texto es muy grande y querés acelerar, podés bajarlo, pero nunca mayor o igual que max_tokens porque max_tokens - stride debe ser positivo.
    if max_tokens is None:
        max_tokens = tokenizer_trad.model_max_length

    encoded = tokenizer_trad(texto, return_tensors="pt")["input_ids"][0]
    resultado = []

    for start in range(0, len(encoded), max_tokens - stride):
        end = min(start + max_tokens, len(encoded))
        segmento_ids = encoded[start:end].unsqueeze(0)
        translated_ids = model_trad.generate(segmento_ids)
        texto_traducido = tokenizer_trad.decode(translated_ids[0], skip_special_tokens=True)
        resultado.append(texto_traducido)

        if end == len(encoded):
            break

    return " ".join(resultado)


# ======================================
# 📄 Traducción de documentos en inglés
# ======================================
#Recorre la carpeta de PDFs en inglés (ruta) y carga cada archivo con PyMuPDFLoader.
def traducir_documentos_en_ingles(ruta):
    documentos = []
    for archivo in os.listdir(ruta):
        if archivo.endswith(".pdf"):
            ruta_completa = os.path.join(ruta, archivo)
            loader = PyMuPDFLoader(ruta_completa)
            docs = loader.load()
            for doc in docs:
                texto_original = doc.page_content
                texto_traducido = traducir_texto(texto_original)
                doc.page_content = texto_traducido
                documentos.append(doc)
    return documentos

# ======================================
# 📦 Ejecutamos la carga total del corpus
# ======================================
documentos_es = cargar_documentos_desde_carpeta([carpetas_es])
documentos_en = traducir_documentos_en_ingles(carpeta_en)

documentos = documentos_es + documentos_en#aqui se juntan los documentos en español y los de inglés ya traducidos al español

"""CHUNKING (División en fragmentos)"""
"""ESte fragmento lo estoy reemplazando porque sólo valida el primer archivo 

# 📌 VALIDACIONES previas antes del chunking

print("🔎 Validando el corpus final...")

# 1. ¿Qué tipo de objeto es el corpus?
print(f"Tipo de corpus_completo: {type(documentos)}")

# 2. ¿Cuántos documentos contiene?
print(f"📚 Total de documentos: {len(documentos)}")

# 3. ¿Qué tipo de objeto es cada documento?
if documentos:
    print(f"Ejemplo de tipo de documento: {type(documentos[0])}")

# 4. Mostrar los primeros 500 caracteres del primer documento
if documentos and hasattr(documentos[0], "page_content"):
    print("\n📝 Vista previa del primer documento:")
    print(documentos[0].page_content[:500])
else:
    print("⚠️ No se encontró texto en el primer documento.")"""
# 📌 VALIDACIONES previas antes del chunking

print("🔎 Validando el corpus español...")
print(f"📚 Total documentos ES: {len(documentos_es)}")
if documentos_es and hasattr(documentos_es[0], "page_content"):
    print(documentos_es[0].page_content[:500])

print("\n🔎 Validando el corpus traducido...")
print(f"📚 Total documentos EN->ES: {len(documentos_en)}")
if documentos_en and hasattr(documentos_en[0], "page_content"):
    print(documentos_en[0].page_content[:500])

documentos = documentos_es + documentos_en
print("\n🔎 Validando el corpus final...")
print(f"Tipo de corpus_completo: {type(documentos)}")
print(f"📚 Total de documentos: {len(documentos)}")

for i, doc in enumerate(documentos[:5], 1):
    nombre = doc.metadata.get("source", f"Documento {i}")
    print(f"\n📝 Vista previa {i} ({nombre}):")
    print(doc.page_content[:500])

"""CHUNKING (División en fragmentos)"""

# ✅ Importamos la herramienta para dividir documentos en fragmentos/chunks

#from langchain.text_splitter import RecursiveCharacterTextSplitter
# Cambio a nueva libreria, langchain.text_splitter me daba errores por versiones de langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ✅ Configuramos el splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Cantidad máxima de caracteres por fragmento
    chunk_overlap=50,      # Cuántos caracteres se solapan entre fragmentos
    separators=["\n\n", "\n", ".", " ", ""],  # Orden de preferencia para cortar texto
)

"""Si los textos son muy técnicos o largos, evaluá en tokens: algunos modelos funcionan mejor con 200-300 tokens 
(~700-900 caracteres), así que podrías subir chunk_size o usar TokenTextSplitter para que el corte sea por tokens
 reales.
Ajustá chunk_overlap según la densidad de información: si cada párrafo es clave, subilo (80-100) para que el 
retriever no pierda contexto; si son textos redundantes, bajarlo acelera el pipeline."""

# ✅ Aplicar el splitter a todos los documentos
chunks = text_splitter.split_documents(documentos)

from pathlib import Path

# ✅ Enriquecer y simplificar la metadata de cada chunk
for chunk in chunks:
    metadata = chunk.metadata

    # Extraer nombre de archivo limpio
    file_name = Path(metadata.get("file_path", "desconocido")).stem

    # Agregar nombre simple
    metadata["nombre_documento"] = file_name

    # Agregar categoría manual según carpeta origen
    if "corpus_base" in metadata.get("file_path", ""):
        metadata["categoria"] = "base_tecnica"
        metadata["idioma"] = "es"
    elif "complementos_tecnicos" in metadata.get("file_path", ""):
        metadata["categoria"] = "complemento"
        metadata["idioma"] = "en"
    else:
        metadata["categoria"] = "otro"
        metadata["idioma"] = "desconocido"

    # Limpiar metadata innecesaria
    for campo in ["producer", "creator", "format", "encryption", "trapped", "moddate", "creationdate", "title", "author", "subject", "keywords"]:
        metadata.pop(campo, None)

# Validar resultados del chunking
print(f"📄 Total de chunks generados: {len(chunks)}")
print("\n📝 Primer chunk:")
print(chunks[0].page_content[:500])

# Revisar la metadata de los primeros 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n🧾 Chunk {i+1} - Metadata:")
    print(chunk.metadata)

"""EMBEDDINGS"""

# ==========================
# Paso 3.1: Cargar modelo de embeddings en español
# ==========================

# Original
# from langchain.embeddings import HuggingFaceEmbeddings
# Nueva
from langchain_huggingface import HuggingFaceEmbeddings
#HuggingFaceEmbeddings es la clase que te permite convertir cada chunk en un vector numérico usando un modelo de Hugging Face
# Definir el modelo de embeddings
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ===============================================
# 🧠 Vectorización de Chunks con ChromaDB
# ===============================================

# Ruta donde se almacenará la base vectorial
persistencia_vectores = "db_vectores"

# Crear base de datos vectorial con los embeddings
chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=modelo_embeddings,
    persist_directory="db_vectores"
)

# Guardar la base persistente en disco
chroma_db.persist()

# Validación visual
print("✅ Base de datos vectorial creada con éxito y guardada en:", persistencia_vectores)

# ====================================================
# 🧠 Cargar base vectorial persistente y preparar el Retriever
# ====================================================


# 🔄 Ruta donde guardaste la base de datos vectorial
persistencia_vectores = "db_vectores"

# 🗃️ Cargar la base vectorial persistente desde el disco
chroma_db = Chroma(
    persist_directory=persistencia_vectores,
    embedding_function=modelo_embeddings
)

# 🔍 Crear un Retriever para realizar búsquedas por similitud
retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

# ✅ Validación
print("Retriever creado correctamente. Listo para recuperar chunks similares.")

# =============================================
# 🟪 Consulta de Prueba y Recuperación (SIN RetrievalQA)
# =============================================


chroma_db = Chroma(
    persist_directory="db_vectores",
    embedding_function=modelo_embeddings
)
# 🔍 Creamos el retriever (mecanismo de recuperación)
retriever = chroma_db.as_retriever(
    search_type="similarity",  # También puedes usar "mmr" (Maximal Marginal Relevance)
    search_kwargs={"k": 3}      # Número de documentos más similares que queremos recuperar
)
# 🧪 Definimos una pregunta de prueba
pregunta_prueba = "¿Qué es fractura hidráulica?"
# Recuperamos los documentos más relevantes usando el retriever directamente

#### GBG - Se intento con esta linea, pero da error, sugiriendo usar una funcion
#### privada que no siempre funciona, y corremos riesgo de que rompa al mandarlo a huggingface
#documentos_recuperados = retriever.get_relevant_documents(pregunta_prueba)

#### Prueba con otro metodo (invoke)
documentos_recuperados = retriever.invoke(pregunta_prueba)

# Mostramos los resultados (esto es parte del "Retrieval", antes de la "Generation")
print("📌 Resultados de la recuperación:")
for i, doc in enumerate(documentos_recuperados, 1):
    print(f"🔹 Documento {i}:")
    print(doc.page_content[:500])  # Muestra los primeros 500 caracteres
    print("📎 Metadata:", doc.metadata)
    print("-" * 80)

# 🧠 Aquí termina la parte de "Retrieval" del RAG.
# La parte de "Generation" (usar un LLM para responder la pregunta con el contexto recuperado)
# se haría en otro paso, por ejemplo, en app.py, usando el retriever y un modelo de lenguaje.
# Por ejemplo, podrías pasar 'documentos_recuperados' y 'pregunta_prueba' a una cadena RAG allí.
print("Se han recuperado los documentos relevantes.")
