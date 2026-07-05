# Lucida - Lesson Clarity Engine

Lucida parses a lesson (pasted text or an uploaded PDF/DOCX) into ordered concept chunks with a Groq LLM, embeds them with sentence-transformers, and generates a synthetic student population to simulate how well the lesson lands. Built as a two-person hackathon project: this repo holds the lesson-parsing backend and the browser frontend; `person_b/` holds the companion comprehension-simulation service.

## Structure

```
Lucida/
├── main.py               # FastAPI backend: GET /health, POST /parse-lesson
├── lesson_parser.py      # Groq LLM lesson -> concept chunks
├── embedder.py           # sentence-transformers embeddings
├── student_generator.py  # synthetic student population
├── models.py             # Pydantic request/response models
├── index.html            # Single-file frontend (open directly in a browser)
├── config.example.js     # Template for frontend API key config
└── person_b/             # Companion simulation service (separate FastAPI app)
```

## Setup

### Backend

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # PowerShell: $env:GROQ_API_KEY="your_key_here"
uvicorn main:app --reload --port 8000
```

### Frontend

The frontend calls Groq directly from the browser and reads its key from a gitignored `config.js`:

```bash
cp config.example.js config.js
# then paste your Groq API key into config.js
```

Open `index.html` in a browser. Never commit `config.js`.

## API

- `GET /health` - status check
- `POST /parse-lesson` - lesson text in, ordered concept chunks + embeddings + student handoff out
