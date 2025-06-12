from google_crc32c import value
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey
from internal.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, unique=True, index=True, autoincrement="auto")
    openweb_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    role = Column(String)

class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, unique=True, index=True, autoincrement="auto")
    title = Column(String)
    value = Column(Integer, unique=True, index=True)

class Config(Base):
    __tablename__ = "configs"

    id = Column(Integer, primary_key=True, unique=True, index=True, autoincrement="ignore_fk")
    user_id = Column(Integer, ForeignKey("users.id"))
    service_id = Column(Integer, ForeignKey("services.id"))
    chess_path = Column(String)
    doc_directory = Column(String)
    is_quantized = Column(Boolean)
    llm_name = Column(String)
    qa_is_quantized = Column(Boolean)
    qa_llm_name = Column(String)
    rag_retrieve_score_threshold = Column(Float)
    rag_topk = Column(Integer)
    rag_vector_db_path = Column(String)
    same_as_above = Column(Boolean)
    seed = Column(Integer)

class Statistic(Base):
    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, unique=True, index=True, autoincrement="ignore_fk")
    user_id = Column(Integer, ForeignKey("users.id"))
    email = Column(String)
    start_date = Column(DateTime)
    last_date = Column(DateTime)
    average_token_length = Column(Integer)
    query_count = Column(Integer)