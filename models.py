from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ParseLessonRequest(BaseModel):
    text: str
    num_students: int = 500


class ConceptChunk(BaseModel):
    order: int
    text: str
    concepts: List[str]
    prerequisites: List[str]
    embedding: List[float]


class PersonalityModifiers(BaseModel):
    comprehension_bonus: float
    random_dropout_chance: float
    threshold_modifier: float


class Student(BaseModel):
    id: int
    personality: str
    knowledge_vector: List[float]
    dropout_threshold: float
    modifiers: PersonalityModifiers


class HandoffResponse(BaseModel):
    chunks: List[ConceptChunk]
    students: List[Student]
    personality_modifiers: Dict[str, PersonalityModifiers]


class HealthResponse(BaseModel):
    status: str
