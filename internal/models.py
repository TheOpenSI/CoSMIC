from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from internal.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, unique=True, index=True)
    openweb_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    role = Column(String)

    # Relationships
    configs = relationship("Config", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    statistics = relationship("Statistic", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, unique=True, index=True)
    title = Column(String)
    # Removed legacy 'value' column. Migration 20251009_drop_service_value drops it.
    # Keep a hybrid_property for backward compatibility so existing code that
    # still references service.value will receive the primary key as a stand-in.
    description = Column(String(255))  # Short explanation of what the service does
    active = Column(Boolean, default=True)  # Soft-enable/disable flag

    @hybrid_property
    def value(self):  # type: ignore
        """Backward-compatible attribute. Returns id now that 'value' column is gone."""
        return self.id
    description = Column(String(255))
    active = Column(Boolean, default=True)

    # Relationships
    configs = relationship("Config", back_populates="service")

class Config(Base):
    __tablename__ = "configs"

    id = Column(Integer, primary_key=True, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True)
    service_id = Column(Integer, ForeignKey("services.id"), index=True, nullable=True)
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

    # Relationships
    user = relationship("User", back_populates="configs")
    service = relationship("Service", back_populates="configs")

class Statistic(Base):
    __tablename__ = "statistics"

    id = Column(Integer, primary_key=True, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True)
    email = Column(String)
    start_date = Column(DateTime)
    last_date = Column(DateTime)
    average_token_length = Column(Integer)
    query_count = Column(Integer)

    # Relationships
    user = relationship("User", back_populates="statistics")


class LLM(Base):
    __tablename__ = "llms"

    id = Column(Integer, primary_key=True, unique=True, index=True)
    # ID of the model in OpenWebUI `model` table
    openweb_model_id = Column(String, unique=True, index=True)  # FK-ish reference to OpenWebUI model.id
    # Human-readable name from OpenWebUI
    name = Column(String)  # Human friendly name
    # Optional base model reference from OpenWebUI
    base_model_id = Column(String)  # Underlying base model if provided
    # Timestamps from OpenWebUI (epoch seconds)
    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)