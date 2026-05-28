import uuid
from fastapi import APIRouter, UploadFile
from pathlib import Path

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile, memory_type: str = "session", chat_session_id: str = "default", user_id: str = "default"
):

    file_id = uuid.uuid4().hex[:8]

    
    if memory_type == "global_memory":
        save_dir = Path("/app/data/memories/global")
    elif memory_type == "user":
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

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "content_type": file.content_type,
    }
