from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserModel(BaseModel):
    id: int
    email: str
    role: str

class ServiceModel(BaseModel):
    id: int  # Primary key
    title: str  # Human readable service title
    description: Optional[str] = None  # Short description of service purpose
    active: bool = True  # Whether service is currently enabled/visible

class ConfigModel(BaseModel):
    id: int
    user_id: int
    service_id: int
    chess_path: Optional[str] = None
    doc_directory: Optional[str] = None
    is_quantized: bool = False
    llm_name: str
    qa_is_quantized: bool = False
    qa_llm_name: str
    rag_retrieve_score_threshold: float = 0.0
    rag_topk: int = 10
    rag_vector_db_path: Optional[str] = None
    same_as_above: bool = False
    seed: Optional[int] = None
    
class StatisticModel(BaseModel):
    id: int
    user_id: int
    email: str
    start_date: datetime
    last_date: datetime
    average_token_length: int = 0
    query_count: int = 0