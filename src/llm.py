import google.generativeai as genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.vecstore import get_retriever

# Configurar API Key
genai.configure(api_key=GEMINI_API_KEY)

def build_prompt(question: str, docs) -> str:
    role = (
        "Actúa como Ingeniero de petróleo y asistente experto con experiencia en "
        "toda la cadena de valor del petróleo y gas, incluidos reservorios no convencionales.\n"
        "Evalúa el contexto con ese criterio técnico, pero responde únicamente con la información suministrada."
    )

    tone = (
        "Redacta en español claro y didáctico para lectores sin experiencia petrolera. "
        "Estructurá la explicación en párrafos cortos y ejemplos simples."
    )

    fallback = (
        "Si la respuesta no está respaldada por los fragmentos del contexto, di explícitamente: "
        "'La información disponible no permite responder con fiabilidad.'"
    )

    format_rules = (
        "Siempre responde en este formato salvo que la respuesta no esté respaldada por los fragmentos del contexto:\n"
        "1) Resumen directo: (salto de línea) (máximo 2 párrafos).\n"
        "2) Pasos o implicaciones prácticas: (salto de linea) (listado en viñetas).\n"
    )

    context_header = "Contexto técnico disponible:\n"
    context = ""
    for d in docs:
        text = d.page_content.strip()
        src = d.metadata.get("source", "desconocido")
        chunk = d.metadata.get("chunk_id", "?")
        context += f"[Documento: {src} | Chunk: {chunk}]\n{text}\n\n"

    question_block = f"Pregunta del usuario:\n{question.strip()}\n"

    prompt = (
        f"{role}\n"
        f"{tone}\n"
        f"{fallback}\n"
        f"{format_rules}\n\n"
        f"{context_header}{context}\n"
        f"{question_block}\n"
        "Responde siguiendo las reglas anteriores:"
    )

    return prompt




def ask_model(pregunta: str, vectorstore, k=5, return_sources=True):
    retriever = get_retriever(k=k)
    docs = retriever.invoke(pregunta)

    # Armamos prompt
    prompt = build_prompt(pregunta, docs)
    
    # llamar al modelo
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    respuesta = response.text

    if not return_sources:
        return respuesta

    sources = [
        {
            "page_content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "?")
        }
        for d in docs
    ]

    return respuesta, sources
