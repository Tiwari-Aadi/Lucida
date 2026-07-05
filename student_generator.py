import random
from typing import List, Dict, Any

import numpy as np


PERSONALITY_CONFIG: Dict[str, Dict[str, Any]] = {
    "curious": {
        "dropout_threshold": 0.50,
        "modifiers": {
            "comprehension_bonus": 0.10,
            "random_dropout_chance": 0.00,
            "threshold_modifier": -0.05,
        },
        "knowledge_skew": "strong",  # 0.7-1.0
    },
    "distracted": {
        "dropout_threshold": 0.60,
        "modifiers": {
            "comprehension_bonus": 0.00,
            "random_dropout_chance": 0.15,
            "threshold_modifier": 0.00,
        },
        "knowledge_skew": "weak",  # 0.1-0.4
    },
    "anxious": {
        "dropout_threshold": 0.75,
        "modifiers": {
            "comprehension_bonus": 0.00,
            "random_dropout_chance": 0.00,
            "threshold_modifier": 0.15,
        },
        "knowledge_skew": "average",  # 0.4-0.6
    },
    "overconfident": {
        "dropout_threshold": 0.45,
        "modifiers": {
            "comprehension_bonus": 0.05,
            "random_dropout_chance": 0.00,
            "threshold_modifier": -0.10,
        },
        "knowledge_skew": "average",  # tends to overestimate, so moderate actual knowledge
    },
    "average": {
        "dropout_threshold": 0.60,
        "modifiers": {
            "comprehension_bonus": 0.00,
            "random_dropout_chance": 0.00,
            "threshold_modifier": 0.00,
        },
        "knowledge_skew": "average",
    },
}

PERSONALITY_TYPES = list(PERSONALITY_CONFIG.keys())

# Approximate distribution across a class
PERSONALITY_WEIGHTS = [0.15, 0.20, 0.15, 0.10, 0.40]  # sums to 1.0


def _sample_knowledge_vector(skew: str, dim: int) -> np.ndarray:
    """
    Sample a knowledge vector based on the student's academic skew.
    Values are clipped to [0, 1].
    """
    rng = np.random.default_rng()

    if skew == "strong":
        # Beta distribution skewed high
        vec = rng.beta(a=5.0, b=2.0, size=dim)
    elif skew == "weak":
        # Beta distribution skewed low
        vec = rng.beta(a=2.0, b=5.0, size=dim)
    else:
        # Roughly symmetric around 0.5
        vec = rng.beta(a=4.0, b=4.0, size=dim)

    return np.clip(vec, 0.0, 1.0)


def generate_students(num_students: int, embedding_dim: int) -> List[Dict[str, Any]]:
    """
    Generate a population of synthetic students.

    Args:
        num_students: how many students to create
        embedding_dim: dimension of each knowledge vector (matches chunk embeddings)

    Returns:
        list of student dicts matching the handoff schema
    """
    personalities = random.choices(
        PERSONALITY_TYPES,
        weights=PERSONALITY_WEIGHTS,
        k=num_students,
    )

    students = []
    for idx, personality in enumerate(personalities):
        config = PERSONALITY_CONFIG[personality]
        knowledge_vec = _sample_knowledge_vector(config["knowledge_skew"], embedding_dim)

        students.append(
            {
                "id": idx + 1,
                "personality": personality,
                "knowledge_vector": knowledge_vec.tolist(),
                "dropout_threshold": config["dropout_threshold"],
                "modifiers": config["modifiers"],
            }
        )

    return students


def get_personality_modifiers() -> Dict[str, Dict[str, Any]]:
    return {
        name: cfg["modifiers"] for name, cfg in PERSONALITY_CONFIG.items()
    }
