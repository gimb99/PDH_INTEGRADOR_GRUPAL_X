# PDH_INTEGRADOR_GRUPAL_X
Integrador Grupal - Tecnicas de Procesamiento del Habla (TPDH) para IFTS24 - 2025

## Descripción
Sistema de Retrieval-Augmented Generation (RAG) que permite a estudiantes consultar información de material didáctico (transcripciones, guías, FAQs) mediante búsqueda semántica y generación de respuestas.

## Demo
[Link a aplicación desplegada o instrucciones para ejecución local]

## Problema que Resuelve
Facilita la búsqueda de información específica en grandes volúmenes de material educativo, permitiendo preguntas en lenguaje natural y obteniendo respuestas contextualizadas con fuentes.

## Arquitectura del Sistema
### Pipeline RAG
1. **Ingesta**: Carga de documentos (PDF, TXT).
2. **Chunking**: División en fragmentos de 500 caracteres con overlap.
3. **Embeddings**: Modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. **Almacenamiento**: ChromaDB.
5. **Retrieval**: Búsqueda por similitud (top-k=3).
6. **Generation**: Modelo `google/flan-t5-base` via Hugging Face Inference API.
7. **Interfaz**: Streamlit.

### Diagrama de Flujo
TODO - SOON

## Stack Tecnológico
- **LLM**: google/flan-t5-base (Hugging Face Inference API)
- **Embeddings**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Vector Database**: ChromaDB
- **Orquestación**: LangChain
- **Interfaz**: Streamlit
- **Deployment**: TODO - Hugging Face Spaces/ Local

## Corpus de Documentos
- **Dominio**: Material educativo (transcripciones, guías, FAQs). TODO - Describir mejor
- **Cantidad**: X documentos.
- **Fuente**: TODO
- **Formato**: [PDF, TXT, etc.].
- **Idioma**: Español.

## Instalación y Uso Local
### Pre-requisitos
- Python 3.9+
- [Opcional si usas Inference API] Cuenta en Hugging Face con token configurado (`hf auth login`).

### Pasos de Instalación
1. Clonar el repositorio:
   ```bash
   git clone [url-de-tu-repo]
   cd [nombre-del-repo]
   ```
2. Crear entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. [Si es primera vez] Procesar documentos:
   ```bash
   python ingest_documents.py
   ```
5. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```
6. Abrir en navegador (localmente): `http://localhost:8501`

## Estructura del Proyecto
- `app.py`: Aplicación Streamlit principal.
- `ingest_documents.py`: Script de ingesta y procesamiento.
- `config.py`: Configuración de modelos.
- `requirements.txt`: Dependencias.
- `README.md`: Este archivo.
- `data/`: Documentos fuente (no subidos al repo).
- `chroma_db/`: Base de datos vectorial (generada).

## Ejemplos de Consultas
- Definir

## Decisiones de Diseño
- ...

## Limitaciones Conocidas
- ...

## Información
- Trabajo Integrador Grupal
- Integrantes: Gonzalo Barthou, Carmen Marylin Rodriguez, Tamara Peña
- Materia: Técnicas de Procesamiento del Habla
- Institución: IFTS 24
- Año: 2025