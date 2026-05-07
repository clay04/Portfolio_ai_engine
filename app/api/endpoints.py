from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.security import verify_api_key
from app.services.rag_service import query_cv, stream_cv

router = APIRouter()


# Schemas
class HistoryMessage(BaseModel):
    role: str      # "user" atau "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    chat_history: list[HistoryMessage] = Field(
        default=[],
        description="Riwayat percakapan sebelumnya untuk konteks memori AI"
    )

class ChatResponse(BaseModel):
    status: str
    answer: str
    source_documents: list[str]


# Non-streaming
@router.post("/chat-cv", response_model=ChatResponse, tags=["CV Chat"])
async def chat_with_cv(body: ChatRequest, _: str = Depends(verify_api_key)):
    try:
        history = [m.model_dump() for m in body.chat_history]
        return query_cv(body.question, chat_history=history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Streaming
@router.post("/chat-cv/stream", tags=["CV Chat"])
async def chat_with_cv_stream(body: ChatRequest, _: str = Depends(verify_api_key)):
    history = [m.model_dump() for m in body.chat_history]
    return StreamingResponse(
        stream_cv(body.question, chat_history=history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Health
@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "CV AI Engine - Clay Mangeber"}