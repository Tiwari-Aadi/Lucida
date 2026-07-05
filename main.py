import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ParseLessonRequest, HandoffResponse, HealthResponse
from lesson_parser import parse_lesson
from embedder import embed_texts
from student_generator import generate_students, get_personality_modifiers

app = FastAPI(
    title="Lucida - Lesson Clarity Engine",
    description="Lesson parsing, embeddings, and synthetic student generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post("/parse-lesson", response_model=HandoffResponse)
async def parse_lesson_endpoint(request: ParseLessonRequest):
    # Validate required env vars early
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing environment variable: GROQ_API_KEY")

    # Step 1 - parse lesson into concept chunks via llama-3.3-70b-versatile
    try:
        raw_chunks = parse_lesson(request.text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Lesson parsing failed: {exc}")

    if not raw_chunks:
        raise HTTPException(status_code=422, detail="Model returned no chunks for the given lesson text.")

    # Step 2 - embed each chunk's text
    try:
        chunk_texts = [chunk["text"] for chunk in raw_chunks]
        embeddings = embed_texts(chunk_texts)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")

    embedding_dim = len(embeddings[0])

    # Attach embeddings to chunks
    chunks_with_embeddings = []
    for chunk, embedding in zip(raw_chunks, embeddings):
        chunks_with_embeddings.append(
            {
                "order": chunk["order"],
                "text": chunk["text"],
                "concepts": chunk["concepts"],
                "prerequisites": chunk["prerequisites"],
                "embedding": embedding,
            }
        )

    # Step 3 - generate synthetic student population
    students = generate_students(request.num_students, embedding_dim)

    # Step 4 - assemble handoff payload
    return {
        "chunks": chunks_with_embeddings,
        "students": students,
        "personality_modifiers": get_personality_modifiers(),
    }
