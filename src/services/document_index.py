### Core modules ###
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

### Type hints ###

### Internal modules ###
from .base import ServiceBase


class DocumentMetadata:
    """Metadata for indexed documents"""
    def __init__(
        self,
        document_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        file_name: str = "",
        memory_type: str = "session",
        chunk_count: int = 0,
        upload_date: Optional[str] = None
    ):
        self.document_id = document_id
        self.user_id = user_id
        self.session_id = session_id
        self.file_name = file_name
        self.memory_type = memory_type
        self.chunk_count = chunk_count
        self.upload_date = upload_date or datetime.now(tz=ZoneInfo("Australia/Sydney")).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "file_name": self.file_name,
            "memory_type": self.memory_type,
            "chunk_count": self.chunk_count,
            "upload_date": self.upload_date
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        return cls(**data)


class DocumentIndex(ServiceBase):
    """In-memory document metadata index with optional persistence"""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize document index.

        Args:
            persist_path (str, optional): Path to persist index as JSON. Defaults to None.
        """
        super().__init__(**kwargs)

        self.documents: Dict[str, DocumentMetadata] = {}
        self.persist_path: Optional[Path] = None

        if persist_path:
            self.persist_path = Path(persist_path).resolve()
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def add_document(
        self,
        document_id: str,
        user_id: str,
        file_name: str,
        memory_type: str = "session",
        session_id: Optional[str] = None,
        chunk_count: int = 0
    ) -> DocumentMetadata:
        """
        Add document metadata to index.

        Args:
            document_id (str): Unique document identifier
            user_id (str): User who uploaded the document
            file_name (str): Original file name
            memory_type (str): Type of memory (global_memory, user, session)
            session_id (str, optional): Associated chat session ID
            chunk_count (int): Number of chunks created from document

        Returns:
            DocumentMetadata: The indexed metadata
        """
        metadata = DocumentMetadata(
            document_id=document_id,
            user_id=user_id,
            session_id=session_id,
            file_name=file_name,
            memory_type=memory_type,
            chunk_count=chunk_count
        )

        self.documents[document_id] = metadata
        self._save_to_disk()

        return metadata

    def get_document_info(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document"""
        return self.documents.get(document_id)

    def get_documents_by_filter(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None
    ) -> List[DocumentMetadata]:
        """
        Retrieve documents matching filter criteria.

        Args:
            user_id (str, optional): Filter by user
            session_id (str, optional): Filter by session
            memory_type (str, optional): Filter by memory type

        Returns:
            List[DocumentMetadata]: Matching documents
        """
        results = []

        for doc in self.documents.values():
            # Apply filters
            if user_id and doc.user_id != user_id:
                continue
            if session_id and doc.session_id != session_id:
                continue
            if memory_type and doc.memory_type != memory_type:
                continue

            results.append(doc)

        # Sort by upload date (newest first)
        results.sort(key=lambda x: x.upload_date, reverse=True)

        return results

    def get_documents_by_user(self, user_id: str) -> List[DocumentMetadata]:
        """Get all documents for a user"""
        return self.get_documents_by_filter(user_id=user_id)

    def get_documents_by_session(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents for a session"""
        return self.get_documents_by_filter(session_id=session_id)

    def remove_document(self, document_id: str) -> bool:
        """
        Remove document from index.

        Args:
            document_id (str): Document to remove

        Returns:
            bool: True if document was removed, False if not found
        """
        if document_id in self.documents:
            del self.documents[document_id]
            self._save_to_disk()
            return True
        return False

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about indexed documents.

        Args:
            user_id (str, optional): Get stats for specific user only

        Returns:
            Dict with statistics
        """
        docs = (
            self.get_documents_by_user(user_id)
            if user_id
            else list(self.documents.values())
        )

        total_chunks = sum(doc.chunk_count for doc in docs)
        docs_by_type = {}

        for doc in docs:
            memory_type = doc.memory_type
            if memory_type not in docs_by_type:
                docs_by_type[memory_type] = {"count": 0, "chunks": 0}
            docs_by_type[memory_type]["count"] += 1
            docs_by_type[memory_type]["chunks"] += doc.chunk_count

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents_by_type": docs_by_type
        }

    def _save_to_disk(self) -> None:
        """Persist index to disk"""
        if not self.persist_path:
            return

        data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self.documents.items()
        }

        with self.persist_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """Load index from disk"""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with self.persist_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = {
                    doc_id: DocumentMetadata.from_dict(metadata)
                    for doc_id, metadata in data.items()
                }
        except Exception as e:
            print(f"Error loading document index from {self.persist_path}: {e}")
            self.documents = {}
