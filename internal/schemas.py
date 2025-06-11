from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserModel(BaseModel):
    id: str
    email: str
    role: str

class ServiceModel(BaseModel):
    id: str
    title: str

class ConfigModel(BaseModel):
    id: str
    user_id: str
    service_id: str
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
    id: str
    user_id: str
    email: str
    start_date: datetime
    last_date: datetime
    average_token_length: int = 0
    query_count: int = 0