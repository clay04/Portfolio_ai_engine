# 🤖 CV AI Engine – Clay Mangeber

AI Engine berbasis **RAG (Retrieval-Augmented Generation)** yang menjawab pertanyaan
tentang Clay Mangeber dari file PDF CV-nya. Dibangun dengan FastAPI + LangChain + Gemini.

---

## 🗂️ Struktur Folder

```
ai-engine/
├── data/
│   └── CV_Clay Aiken mangeber jr.pdf   ← Taruh CV kamu di sini
├── app/
│   ├── main.py                          ← Entry point & Swagger config
│   ├── core/
│   │   ├── config.py                    ← Settings & env variables
│   │   └── security.py                  ← X-API-KEY validation
│   ├── services/
│   │   └── rag_service.py               ← Logika RAG (LangChain, FAISS, Gemini)
│   └── api/
│       └── endpoints.py                 ← Route POST /v1/chat-cv
├── .env                                 ← API Keys (JANGAN di-commit ke Git!)
├── .env.example                         ← Template .env
├── requirements.txt
└── README.md
```

---

## ⚡ Cara Setup & Menjalankan

### 1. Clone & Masuk ke Folder
```bash
cd ai-engine
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
pip install pydantic-settings    # Tambahan untuk config management
```

### 4. Setup File .env
```bash
cp .env.example .env
# Buka .env dan isi GOOGLE_API_KEY dan INTERNAL_API_KEY
```

### 5. Taruh File CV
Pastikan file PDF ada di path ini (sesuaikan nama jika berbeda):
```
data/CV_Clay Aiken mangeber jr.pdf
```

### 6. Jalankan Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 Akses & Testing

| URL | Deskripsi |
|-----|-----------|
| `http://localhost:8000/docs` | **Swagger UI** – Test langsung di browser |
| `http://localhost:8000/redoc` | ReDoc – Dokumentasi alternatif |
| `http://localhost:8000/v1/health` | Health check (tidak perlu API key) |

### Test via curl
```bash
curl -X POST http://localhost:8000/v1/chat-cv \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: kunci-rahasia-kamu" \
  -d '{"question": "Apa IPK Clay saat lulus dari Unklab?"}'
```

### Contoh Response
```json
{
  "status": "success",
  "answer": "Clay lulus dari Universitas Klabat pada Januari 2026 dengan IPK 3.73 (Magna Cum Laude).",
  "source_documents": ["CV_Clay Aiken mangeber jr.pdf"]
}
```

---

## 🔗 Integrasi dengan Backend Node.js

Tambahkan header `X-API-KEY` di setiap request dari Node.js:

```javascript
// axios example
const response = await axios.post('http://localhost:8000/v1/chat-cv', 
  { question: userMessage },
  { headers: { 'X-API-KEY': process.env.AI_ENGINE_KEY } }
);
```

---

## 🔄 Update CV

Cukup ganti file PDF di folder `data/` lalu **restart server**.
Tidak perlu training ulang — FAISS akan di-rebuild otomatis saat startup.

---

## 🛡️ Keamanan

- `GOOGLE_API_KEY` disimpan di `.env` (tidak pernah di-hardcode)
- `.env` **wajib** masuk ke `.gitignore`
- `X-API-KEY` memastikan hanya backend Node.js yang bisa mengakses engine ini
