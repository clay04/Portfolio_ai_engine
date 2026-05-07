import os
import time
import json
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

# Konstanta
FAISS_INDEX_PATH = "data/faiss_index"

# Error Gemini yang memicu fallback ke Groq
GEMINI_FALLBACK_ERRORS = (
    "429",           # Rate limit / quota habis
    "quota",         # Quota exceeded
    "RESOURCE_EXHAUSTED",
    "503",           # Service unavailable
    "overloaded",    # Model overloaded
)

# System Prompt
SYSTEM_PROMPT = """Anda adalah asisten virtual profesional untuk Clay Mangeber, \
seorang lulusan Informatika dari Universitas Klabat. \
Tugas Anda adalah menjawab pertanyaan rekruter atau siapapun yang ingin mengetahui lebih \
lanjut tentang kualifikasi, pengalaman, dan proyek Clay.

ATURAN PENTING:
1. Jawablah HANYA berdasarkan dokumen CV yang disediakan di bawah ini.
2. Jika informasi yang ditanyakan TIDAK ADA di dokumen, jawab dengan jujur: \
   "Maaf, informasi tersebut tidak tersedia di CV Clay."
3. Anda MEMILIKI MEMORI percakapan — gunakan riwayat chat untuk menjawab \
   pertanyaan lanjutan.
4. Jawab dengan bahasa yang profesional dan ramah.
5. Gunakan bahasa yang sama dengan bahasa pertanyaan (Indonesia atau Inggris).
6. Jawaban HARUS ringkas, maksimal 3–5 kalimat.
7. Gunakan bullet point jika menjelaskan lebih dari 1 poin.
8. Hindari paragraf panjang.

Konteks dari CV Clay:
{context}"""

# Singleton state 
_retriever = None
_llm_gemini = None
_llm_groq = None


# Custom Embedding Class — Bypass gRPC
class GeminiRESTEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-2"):
        genai.configure(api_key=api_key)
        self.model = model

    def _embed_with_retry(self, text: str, task_type: str, max_retries: int = 3) -> list[float]:
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(model=self.model, content=text, task_type=task_type)
                return result["embedding"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  Embedding retry {attempt+1} dalam {wait}s... Error: {e}")
                time.sleep(wait)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_with_retry(text, "retrieval_document"))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_with_retry(text, "retrieval_query")


# Pipeline Setup
def _load_or_build_vectorstore() -> FAISS:
    embeddings = GeminiRESTEmbeddings(api_key=settings.GOOGLE_API_KEY)

    if os.path.exists(FAISS_INDEX_PATH):
        print("Memuat FAISS index dari disk...")
        vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index dimuat.")
        return vs

    pdf_path = settings.PDF_PATH
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File CV tidak ditemukan: '{pdf_path}'.")

    print(" Membaca PDF...")
    documents = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    ).split_documents(documents)
    print(f"   Total chunks: {len(chunks)}")

    vs = None
    for i, chunk in enumerate(chunks):
        print(f"   Embedding chunk {i+1}/{len(chunks)}...")
        if vs is None:
            vs = FAISS.from_documents([chunk], embeddings)
        else:
            vs.merge_from(FAISS.from_documents([chunk], embeddings))

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index disimpan.")
    return vs


def get_pipeline():
    """Singleton: retriever + kedua LLM (Gemini & Groq), dibuild sekali saat startup."""
    global _retriever, _llm_gemini, _llm_groq
    if _retriever is None:
        print("Membangun RAG pipeline...")
        vs = _load_or_build_vectorstore()
        _retriever = vs.as_retriever(search_kwargs={"k": 4})

        # Primary LLM — Gemini
        _llm_gemini = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
            streaming=True,
        )

        # Fallback LLM — Groq
        _llm_groq = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
            streaming=True,
        )

        print(f"RAG pipeline siap. Primary: {settings.GEMINI_MODEL} | Fallback: {settings.GROQ_MODEL}")
    return _retriever, _llm_gemini, _llm_groq


# Helpers
def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _get_sources(docs) -> list[str]:
    return list({
        os.path.basename(doc.metadata.get("source", "CV_Clay Aiken mangeber jr.pdf"))
        for doc in docs
    })


def _build_messages(question: str, context: str, chat_history: list[dict]) -> list:
    """
    Bangun list messages:
    [SystemMessage] → [history...] → [HumanMessage(question)]
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    for msg in chat_history[-10:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def _should_fallback(error: Exception) -> bool:
    """Cek apakah error dari Gemini perlu fallback ke Groq."""
    error_str = str(error).lower()
    return any(trigger.lower() in error_str for trigger in GEMINI_FALLBACK_ERRORS)


# Public API
def query_cv(question: str, chat_history: list[dict] = []) -> dict:
    """Non-streaming dengan Gemini → Groq fallback."""
    retriever, llm_gemini, llm_groq = get_pipeline()

    docs = retriever.invoke(question)
    context = _format_docs(docs)
    sources = _get_sources(docs)
    messages = _build_messages(question, context, chat_history)

    # Coba Gemini dulu 
    try:
        print(f"🤖  Menggunakan {settings.GEMINI_MODEL}...")
        response = llm_gemini.invoke(messages)
        return {
            "status": "success",
            "answer": response.content,
            "source_documents": sources,
            "model_used": settings.GEMINI_MODEL,
        }
    except Exception as e:
        if not _should_fallback(e):
            raise  # Error lain (bukan quota/rate limit) — langsung raise

        print(f"Gemini gagal ({e}). Fallback ke {settings.GROQ_MODEL}...")

    # Fallback ke Groq 
    try:
        response = llm_groq.invoke(messages)
        return {
            "status": "success",
            "answer": response.content,
            "source_documents": sources,
            "model_used": settings.GROQ_MODEL,  # Info model yang dipakai
        }
    except Exception as e:
        raise RuntimeError(f"Semua LLM gagal. Error Groq: {e}")


def stream_cv(question: str, chat_history: list[dict] = []):
    """
    Streaming generator dengan Gemini → Groq fallback.

    SSE events:
      {"type": "model",   "data": "gemini-2.5-flash"}   ← model yang dipakai
      {"type": "sources", "data": ["file.pdf"]}
      {"type": "token",   "data": "Clay"}
      {"type": "done"}
      {"type": "error",   "data": "pesan error"}
    """
    try:
        retriever, llm_gemini, llm_groq = get_pipeline()

        docs = retriever.invoke(question)
        context = _format_docs(docs)
        sources = _get_sources(docs)
        messages = _build_messages(question, context, chat_history)

        # Tentukan LLM yang akan dipakai 
        llm_to_use = llm_gemini
        model_name = settings.GEMINI_MODEL

        yield f"data: {json.dumps({'type': 'model', 'data': model_name})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Stream dengan Gemini
        try:
            for chunk in llm_to_use.stream(messages):
                token = chunk.content
                if token:
                    escaped = token.replace("\n", "\\n")
                    yield f"data: {json.dumps({'type': 'token', 'data': escaped})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return  

        except Exception as e:
            if not _should_fallback(e):
                raise  # Error lain — langsung raise ke outer try

            print(f"Gemini stream gagal ({e}). Fallback ke {settings.GROQ_MODEL}...")

            # Beri tahu client bahwa kita switch model
            yield f"data: {json.dumps({'type': 'model', 'data': settings.GROQ_MODEL})}\n\n"

        # Fallback: Stream dengan Groq
        for chunk in llm_groq.stream(messages):
            token = chunk.content
            if token:
                escaped = token.replace("\n", "\\n")
                yield f"data: {json.dumps({'type': 'token', 'data': escaped})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"