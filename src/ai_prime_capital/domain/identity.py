from hashlib import sha256
from json import dumps
from uuid import NAMESPACE_URL, uuid5


def document_id(content: bytes) -> str:
    if not content:
        raise ValueError("document content must not be empty")
    
    return f"doc_{sha256(content).hexdigest()}"


def chunk_id(
    document_id_value: str,
    schema_version: str,
    chunk_index: int,
) -> str:
    if not document_id_value.strip():
        raise ValueError("document_id must not be blank")
    if not schema_version.strip():
        raise ValueError("schema_version must not be blank")
    if chunk_index < 0:
        raise ValueError("chunk_index must not be negative")
    
    identity = dumps(
        [document_id_value, schema_version, chunk_index],
        separators=(",", ":"),
    )
    
    return str(uuid5(NAMESPACE_URL, f"ai-prime-capital/chunk/{identity}"))
