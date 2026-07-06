# app/services/storage.py
import uuid
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_suffix(filename: str) -> str:
    """Take only the extension from the client-supplied filename, never the
    full name/path, and validate it against an allow-list."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}"
            ),
        )
    return suffix


async def save_file(file) -> str:
    suffix = _safe_suffix(file.filename)

    content = await file.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_MB}MB",
        )

    # File name is fully server-generated — the original name is never
    # used to build a path, eliminating path-traversal risk.
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{suffix}"

    with open(file_path, "wb") as f:
        f.write(content)

    return str(file_path)
