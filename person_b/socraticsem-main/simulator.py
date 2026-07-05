import numpy as np
from typing import List, Dict, Any

# Personality modifiers — these come from Person A's student DNA
PERSONALITY_MODIFIERS = {
    "curious": {
        "comprehension_bonus": 0.1,
        "dropout_chance": 0.0,
        "prerequisite_threshold": 0.3,  # more forgiving
        "recovery_rate": 0.15,          # can recover faster
    },
    "distracted": {
        "comprehension_bonus": 0.0,
        "dropout_chance": 0.15,         # random dropout per chunk
        "prerequisite_threshold": 0.5,
        "recovery_rate": 0.0,
    },
    "anxious": {
        "comprehension_bonus": 0.0,
        "dropout_chance": 0.05,
        "prerequisite_threshold": 0.7,  # needs strong foundation
        "recovery_rate": 0.05,
    },
    "overconfident": {
        "comprehension_bonus": 0.05,
        "dropout_chance": 0.08,         # skips over gaps
        "prerequisite_threshold": 0.2,  # thinks they know more than they do
        "recovery_rate": 0.0,
    },
    "average": {
        "comprehension_bonus": 0.0,
        "dropout_chance": 0.0,
        "prerequisite_threshold": 0.5,
        "recovery_rate": 0.05,
    },
}


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_chunk_comprehension(
    student: Dict[str, Any],
    chunk: Dict[str, Any],
    personality: Dict[str, Any],
) -> float:
    """
    Core comprehension function:
    - Cosine similarity between student knowledge vector and chunk concept vector
    - Apply personality modifiers
    - Returns comprehension score 0.0 to 1.0
    """
    # Base comprehension from vector similarity
    base = cosine_similarity(
        student["knowledge_vector"],
        chunk["concept_vector"]
    )

    # Apply personality bonus
    score = base + personality["comprehension_bonus"]

    # Random dropout for distracted students
    if np.random.random() < personality["dropout_chance"]:
        score *= 0.3  # severe attention drop

    # Anxious students need higher prerequisite confidence
    if student["comprehension_score"] < personality["prerequisite_threshold"]:
        score *= 0.5  # struggling to keep up compounds anxiety

    return float(np.clip(score, 0.0, 1.0))


def degrade_knowledge_vector(
    knowledge_vector: List[float],
    chunk_concept_vector: List[float],
    comprehension: float,
    decay_rate: float = 0.15
) -> List[float]:
    """
    CASCADE EFFECT:
    When a student fails to understand a chunk, their knowledge vector
    degrades in the direction of that chunk's concepts — they enter
    the next chunk weaker than before.

    The lower the comprehension, the stronger the degradation.
    """
    kv = np.array(knowledge_vector)
    cv = np.array(chunk_concept_vector)

    # How much to degrade — inversely proportional to comprehension
    degradation_strength = decay_rate * (1.0 - comprehension)

    # Pull knowledge vector away from the concept vector they failed on
    kv = kv - degradation_strength * cv

    # Normalize to keep vector on unit sphere
    norm = np.linalg.norm(kv)
    if norm > 0:
        kv = kv / norm

    return kv.tolist()


def run_simulation(
    chunks: List[Dict[str, Any]],
    students: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Main simulation loop.

    Args:
        chunks: List of chunk dicts from Person A's parse_lesson()
                Each chunk: { id, text, concept_vector, complexity, key_concepts }
        students: List of student dicts from Person A's generate_students()
                  Each student: { id, knowledge_vector, personality_type, ... }

    Returns:
        results_json with comprehension curve, dropout cliffs, per-chunk stats
    """

    n_chunks = len(chunks)
    n_students = len(students)

    # Track per-chunk stats
    chunk_stats = []

    # Track each student's journey
    student_journeys = []

    # Deep copy students so we can mutate their state during simulation
    import copy
    active_students = copy.deepcopy(students)

    # Initialize student state
    for student in active_students:
        student["comprehension_score"] = 1.0   # start fully engaged
        student["state"] = "engaged"           # engaged / struggling / lost
        student["lost_at_chunk"] = -1          # -1 = never lost
        student["journey"] = []                # comprehension per chunk

    for chunk_idx, chunk in enumerate(chunks):
        engaged_count = 0
        struggling_count = 0
        lost_count = 0
        chunk_comprehension_scores = []

        for student in active_students:
            # Already lost — cascade keeps them lost
            if student["state"] == "lost":
                student["journey"].append(0.0)
                lost_count += 1
                continue

            personality = PERSONALITY_MODIFIERS[student["personality_type"]]

            # Compute comprehension for this chunk
            chunk_comp = compute_chunk_comprehension(student, chunk, personality)

            # CASCADE: degrade knowledge vector if comprehension is low
            if chunk_comp < 0.5:
                student["knowledge_vector"] = degrade_knowledge_vector(
                    student["knowledge_vector"],
                    chunk["concept_vector"],
                    chunk_comp
                )

            # Update rolling comprehension score (weighted recent)
            student["comprehension_score"] = (
                0.65 * student["comprehension_score"] +
                0.35 * chunk_comp
            )

            # State transitions
            if student["comprehension_score"] < 0.25:
                if student["state"] != "lost":
                    student["state"] = "lost"
                    student["lost_at_chunk"] = chunk_idx
                lost_count += 1
            elif student["comprehension_score"] < 0.5:
                student["state"] = "struggling"
                struggling_count += 1
            else:
                student["state"] = "engaged"
                engaged_count += 1

            student["journey"].append(round(student["comprehension_score"], 3))
            chunk_comprehension_scores.append(student["comprehension_score"])

        # Compute chunk-level stats
        total = n_students
        avg_comprehension = float(np.mean(chunk_comprehension_scores)) if chunk_comprehension_scores else 0.0

        chunk_stats.append({
            "chunk_index": chunk_idx,
            "chunk_text": chunk["text"],
            "key_concepts": chunk.get("key_concepts", []),
            "engaged_pct": round(engaged_count / total * 100, 1),
            "struggling_pct": round(struggling_count / total * 100, 1),
            "lost_pct": round(lost_count / total * 100, 1),
            "avg_comprehension": round(avg_comprehension, 3),
            "cumulative_lost": lost_count,
        })

    # Build student journeys summary
    for student in active_students:
        student_journeys.append({
            "id": student["id"],
            "personality_type": student["personality_type"],
            "lost_at_chunk": student["lost_at_chunk"],
            "final_comprehension": student["comprehension_score"],
            "journey": student["journey"],
        })

    return {
        "n_students": n_students,
        "n_chunks": n_chunks,
        "chunk_stats": chunk_stats,
        "student_journeys": student_journeys,
    }
