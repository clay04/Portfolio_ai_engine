from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    INTERNAL_API_KEY: str
    PDF_PATH: str = "data/CV_Clay Aiken mangeber jr.pdf"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    GEMINI_MODEL: str = "gemini-2.5-flash"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"

settings = Settings()