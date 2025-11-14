"""
Este directorio se usa para agregar funciones exploradas pero
que al ejecutar en ambientes locales terminaron generando problemas 
de ejecucion. Sea por temas de versiones, o porque las funciones
ya no son soportadas al usar un import de ellas.

### En resumidas cuentas, ES CODIGO QUE NO DEBERIA EJECUTARSE,
pero nos funciona como un archivado para ir limpiando 
"""

# =============================================
# 🟪 Paso 5.2: Consulta de Prueba y Recuperación
# =============================================
# Ya importada
#from langchain_community.vectorstores import Chroma

## GBG - Estos imports van a fallar si usas requirements.txt
""" from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document 
from langchain_community.chains import RetrievalQA
 """

# Pruebo con alternativa para RetrievalQA
#from langchain.chains.retrieval import RetrievalQA
## Esta linea de arriba me da problemas, voy a tener que rehacer RETRIEVAL con otra libreria
## RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 🧠 Cargamos la base vectorial persistida
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
pregunta_prueba = "¿Qué técnicas se utilizan en el fracturamiento hidráulico de reservorios no convencionales?"

# 🔄 Recuperamos los documentos más relevantes
resultados = retriever.get_relevant_documents(pregunta_prueba)

# 🖨️ Mostramos los resultados
print("📌 Resultados de la recuperación:\n")
for i, doc in enumerate(resultados, 1):
    print(f"🔹 Documento {i}:")
    print(doc.page_content[:500])  # Muestra los primeros 500 caracteres
    print("📎 Metadata:", doc.metadata)
    print("-" * 80)

####################################
####################################
