from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np

from simulator import run_simulation
from analyzer import analyze_results


def _l2_normalize(vec: list) -> list:
    """L2-normalize a vector onto the unit sphere."""
    a = np.array(vec, dtype=float)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else a.tolist()


def _abs_normalize(vec: list) -> list:
    """Take absolute values then L2-normalize (maps mixed-sign embeddings to positive unit sphere)."""
    a = np.abs(np.array(vec, dtype=float))
    n = np.linalg.norm(a)
    return (a / n).tolist() if n > 0 else a.tolist()

app = FastAPI(title="Lucida API", version="1.0.0")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    lesson_text: str
    n_students: Optional[int] = 500

class SimulateResponse(BaseModel):
    success: bool
    results: dict


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Lucida"}


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest):
    """
    Full pipeline:
    1. Parse lesson into chunks (Person A's parse_lesson)
    2. Generate students (Person A's generate_students)
    3. Run simulation with cascade effects (Person B's run_simulation)
    4. Analyze results + generate fixes (Person B's analyze_results)
    """
    try:
        # ── Person A's functions (import when their code is ready) ──
        # from parser import parse_lesson
        # from student_generator import generate_students
        # chunks = parse_lesson(request.lesson_text)
        # students = generate_students(request.n_students)

        # ── TEMP: mock Person A's output for testing while they build ──
        chunks, students = mock_person_a(request.lesson_text, request.n_students)

        # ── Person B: Run simulation ──
        simulation_results = run_simulation(chunks, students)

        # ── Person B: Analyze results ──
        analysis = analyze_results(simulation_results)

        return SimulateResponse(success=True, results=analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/chunks-only")
async def simulate_from_chunks(payload: dict):
    """
    Accepts Person A's handoff JSON directly.
    Adapts Person A's schema to Person B's internal format before simulation.

    Body: { "chunks": [...], "students": [...] }
    """
    try:
        raw_chunks = payload.get("chunks", [])
        raw_students = payload.get("students", [])

        if not raw_chunks or not raw_students:
            raise HTTPException(status_code=400, detail="chunks and students required")

        # ── Adapt Person A's chunk schema → Person B's expected format ──
        # Person A:  { order, text, concepts, prerequisites, embedding (384-dim, mixed-sign) }
        # Person B:  { id, text, concept_vector, complexity, key_concepts }
        # Embedding is abs-normalized so cosine similarity with knowledge vectors
        # (all-positive Beta samples) falls in [0, 1] rather than collapsing near 0.
        chunks = [
            {
                "id": c.get("order", c.get("id", i)),
                "text": c["text"],
                "concept_vector": _abs_normalize(c.get("embedding", c.get("concept_vector", []))),
                "key_concepts": c.get("concepts", c.get("key_concepts", [])),
                "complexity": min(1.0, len(c["text"].split()) / 30),
            }
            for i, c in enumerate(raw_chunks)
        ]

        # ── Adapt Person A's student schema → Person B's expected format ──
        # Person A:  { id, personality, knowledge_vector (384-dim Beta [0,1]), dropout_threshold, modifiers }
        # Person B:  { id, knowledge_vector, personality_type, ... }
        # Knowledge vector is L2-normalized so cosine similarity is well-defined.
        students = [
            {
                "id": s["id"],
                "knowledge_vector": _l2_normalize(s["knowledge_vector"]),
                "personality_type": s.get("personality", s.get("personality_type", "average")),
            }
            for s in raw_students
        ]

        simulation_results = run_simulation(chunks, students)
        analysis = analyze_results(simulation_results)

        return {"success": True, "results": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── TEMP: Mock Person A output for independent testing ───────────────────────
# DELETE this once Person A's code is ready and swap in real imports above

def mock_person_a(lesson_text: str, n_students: int):
    """
    Temporary mock of Person A's parse_lesson + generate_students.
    Lets Person B test the full pipeline independently.
    Replace with real imports once Person A is done.
    """
    import numpy as np
    import re

    # Split lesson into sentence-level chunks
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', lesson_text) if len(s.strip()) > 20]
    if not sentences:
        sentences = [lesson_text]

    # Mock chunks with random concept vectors
    chunks = []
    for i, sentence in enumerate(sentences):
        chunks.append({
            "id": i,
            "text": sentence,
            "concept_vector": np.random.randn(64).tolist(),  # 64-dim mock embedding
            "complexity": float(np.clip(len(sentence.split()) / 30, 0.1, 1.0)),
            "key_concepts": sentence.split()[:3],  # first 3 words as mock concepts
        })

    # Mock students with random knowledge vectors + personality distribution
    personalities = ["curious", "distracted", "anxious", "overconfident"]
    # Realistic classroom distribution
    personality_weights = [0.25, 0.35, 0.25, 0.15]

    students = []
    for i in range(n_students):
        # Prior knowledge: most students below average (beta distribution)
        knowledge_base = np.random.beta(2, 4)
        knowledge_vector = np.random.randn(64) * knowledge_base
        norm = np.linalg.norm(knowledge_vector)
        if norm > 0:
            knowledge_vector = knowledge_vector / norm

        students.append({
            "id": i,
            "knowledge_vector": knowledge_vector.tolist(),
            "personality_type": np.random.choice(personalities, p=personality_weights),
            "comprehension_score": 1.0,
            "state": "engaged",
            "lost_at_chunk": -1,
            "journey": [],
        })

    return chunks, students


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
