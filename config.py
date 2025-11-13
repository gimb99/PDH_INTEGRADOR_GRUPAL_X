from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings

### Aquí definimos los modelos y configuraciones centrales para reutilizarlos en otros archivos.

# Configuración del LLM (Flan-T5 Base via Inference API)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",  # T5 usa text2text
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
    # token="TU_TOKEN_AQUI",  # Omitir - Ya se deberia haber hecho login por HuggingFace
    # En ambiente local
)

# Configuración de Embeddings (multilingüe)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Directorio donde se guardará la base de datos vectorial
PERSIST_DIRECTORY = "./chroma_db"