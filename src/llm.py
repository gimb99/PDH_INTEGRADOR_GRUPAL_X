import requests
from src.config import HF_API_KEY, GENERATION_MODEL
from src.vecstore import get_retriever

def build_prompt(question: str, docs) -> str:
    """
    Construye un prompt usando k documentos como contexto.
    """
    header = (
        "Eres un asistente experto. Usa SOLO la información del contexto.\n"
        "Si no hay datos suficientes, decí claramente que no tenés la respuesta.\n\n"
    )

    context = ""
    for d in docs:
        text = d.page_content
        src = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk_id", "?")
        context += f"[FUENTE: {src} | CHUNK: {chunk}]\n{text}\n\n"

    prompt = (
        header +
        "CONTEXTO:\n" +
        context +
        "\nPREGUNTA:\n" +
        question +
        "\n\nRESPUESTA:"
    )

    return prompt


def call_hf(prompt: str) -> str:
    """Llama a HuggingFace Inference API."""
    url = f"https://router.huggingface.co/hf-inference/models/{GENERATION_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.3
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    try:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        return str(data)
    except:
        return f"Error del modelo: {response.text}"


def ask_model(pregunta: str, vectorstore, k: int = 5, return_sources=True):
    """
    Flujo RAG completo:
    1) Recupera k documentos relevantes desde Chroma
    2) Construye prompt
    3) Llama al modelo de HF
    4) Devuelve respuesta + fuentes (si return_sources=True)
    """

    retriever = get_retriever(k=k)
    docs = retriever.invoke(pregunta)

    # Crear prompt
    prompt = build_prompt(pregunta, docs)

    # Llamar modelo
    respuesta = call_hf(prompt)

    if not return_sources:
        return respuesta

    # Extraer metadata
    sources = [
        {
            "page_content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "?")
        }
        for d in docs
    ]



    return respuesta, sources