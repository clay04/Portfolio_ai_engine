from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.rag_service import get_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bangun RAG chain sekali saat server nyala agar request pertama tidak lambat."""
    try:
        get_pipeline()
    except FileNotFoundError as e:
        print(f"⚠️  PERINGATAN: {e}")
        print("   Server tetap berjalan, tapi endpoint /chat-cv akan error.")
    yield 
    print("🛑  Server dimatikan.")


app = FastAPI(
    title="CV AI Engine – Clay Mangeber",
    description=(
        "## AI Engine berbasis RAG (Retrieval-Augmented Generation)\n\n"
        "Engine ini menjawab pertanyaan tentang **Clay Mangeber** "
        "berdasarkan data dari file PDF CV-nya.\n\n"
        "### Cara Penggunaan\n"
        "1. Masukkan `X-API-KEY` kamu di tombol **Authorize** (kanan atas).\n"
        "2. Gunakan endpoint `POST /v1/chat-cv` untuk bertanya.\n\n"
        "### Stack\n"
        "- **Framework**: FastAPI\n"
        "- **RAG**: LangChain + FAISS\n"
        "- **LLM**: Google Gemini 2.5 Flash\n"
        "- **Embedding**: text-embedding-2\n"
    ),
    version="1.0.0",
    contact={
        "name": "Clay Aiken Mangeber Jr",
        "email": "clayaikenmangeberjr@gmail.com",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")
