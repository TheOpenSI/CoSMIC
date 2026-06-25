import mimetypes
import os
import uuid
from fastapi import APIRouter, UploadFile, HTTPException, status
from pathlib import Path

from ...src.services.document_index import DocumentMetadata, compute_content_hash

router = APIRouter()

# Embeddable file types
EMBEDDABLE_EXTENSIONS = {".pdf"}

# Global and User document index instances
# These will be set by the cosmic router initialization
_global_document_index = None
_user_document_index = None
# Vector database to embed uploaded files
_vector_database = None


def set_document_indexes(global_index, user_index):
    """Initialize document indexes (called by cosmic router)"""
    global _global_document_index, _user_document_index
    _global_document_index = global_index
    _user_document_index = user_index


def set_vector_database(vector_database):
    """Vector database to embed uploaded files (called at startup)"""
    global _vector_database
    _vector_database = vector_database


@router.post("/upload")
async def upload_file(
    file: UploadFile, memory_type: str = "session", chat_session_id: str = "default", user_id: str = "default"
):
    if memory_type == "global_memory":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Global memory uploads are not supported at the API level."
        )

    # ---- Duplicate filename check ----
    # Reject the upload if the same filename already exists in the index
    # for this user / session / memory_type combination.
    if _user_document_index:
        existing_docs = _user_document_index.get_documents_by_filter(
            user_id=user_id,
            session_id=chat_session_id if memory_type == "session" else None,
            memory_type=memory_type,
        )
        for doc in existing_docs:
            if doc.file_name == file.filename:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A file with the name '{file.filename}' already exists."
                )

    file_id = uuid.uuid4().hex[:8]

    if memory_type == "user":
        save_dir = Path(f"/app/data/memories/users/{user_id}")
    else:  # memory_type == "session"
        save_dir = Path(f"/app/data/memories/users/{user_id}/sessions/{chat_session_id}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # combine path + file name -> example: Path("/app/data/chat_session_memory/123/manile.pdf")
    file_path = save_dir / f"{file_id}_{file.filename}"

    # then read the file
    content = await file.read()

    # then creates folder and open it
    f = open(file_path, "wb")
    # then write content
    f.write(content)
    f.close()

    # Compute content hash for change-detection
    content_hash = compute_content_hash(str(file_path))
    file_ext = os.path.splitext(file.filename)[1].lower()

    # Embed into the vector database with per-tier metadata 
    # so retrieval can be scoped to this user / session
    chunk_count = 0
    if _vector_database is not None and file_ext in EMBEDDABLE_EXTENSIONS:
        payload = DocumentMetadata(
            document_id=file_id,
            user_id=user_id,
            session_id=chat_session_id if memory_type == "session" else None,
            file_name=file.filename,
            memory_type=memory_type,
            content_hash=content_hash,
        ).to_vector_payload()
        try:
            chunk_count = _vector_database.update_database_from_document(
                str(file_path), extra_metadata=payload
            )
        except Exception:
            chunk_count = 0

    # Add to document index if available
    if _user_document_index:
        _user_document_index.add_document(
            document_id=file_id,
            user_id=user_id,
            file_name=file.filename,
            memory_type=memory_type,
            session_id=chat_session_id,
            chunk_count=chunk_count,
            content_hash=content_hash,
            file_size=len(content),
            file_ext=file_ext,
            content_type=file.content_type or mimetypes.guess_type(file.filename)[0],
        )

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "content_type": file.content_type,
        "chunk_count": chunk_count,
    }