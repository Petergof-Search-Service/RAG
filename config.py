import os
from dotenv import load_dotenv

load_dotenv()

def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(f"Переменная окружения {name} должна быть int, сейчас: {value!r}") from e


RAG_BUCKET_NAME = os.getenv("RAG_BUCKET_NAME")
RAG_S3_ENDPOINT_URL = os.getenv("RAG_S3_ENDPOINT_URL")
RAG_MAX_CHUNK_LEN = getenv_int("RAG_MAX_CHUNK_LEN", 8000)
RAG_YANDEX_API_KEY = os.getenv("RAG_YANDEX_API_KEY")
RAG_ACCESS_KEY = os.getenv("RAG_ACCESS_KEY")
RAG_YANDEX_FOLDER_ID = os.getenv("RAG_YANDEX_FOLDER_ID")
RAG_SECRET_KEY = os.getenv("RAG_SECRET_KEY")
RAG_CHUNKS_PATH = os.getenv("RAG_CHUNKS_PATH")
