import os
from pinecone import Pinecone, ServerlessSpec
from embeddings import embed_text, EMBEDDING_DIM

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hr-resume-chunks")

pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index():
    """
    Проверяет, что индекс существует, и при необходимости создаёт его.
    Вызывай один раз при инициализации системы (или руками).
    """
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Индекс {INDEX_NAME} не найден, создаю...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        print(f"Индекс {INDEX_NAME} уже существует.")

def get_index():
    return pc.Index(INDEX_NAME)


def upsert_chunk(vector_id: str, text: str, metadata: dict | None = None):
    """
    Считает эмбеддинг текста и кладёт его в Pinecone под указанным ID.
    metadata – словарь с дополнительной инфой (candidate_id, resume_id, chunk_index и т.п.)
    """
    vec = embed_text(text)
    index = get_index()

    index.upsert(
        vectors=[{
            "id": vector_id,
            "values": vec,
            "metadata": metadata or {}
        }]
    )


def search_similar(query_text: str, top_k: int = 5, filter: dict | None = None):
    """
    Эмбеддит запрос и ищет ближайшие чанки в индексе.
    filter – опциональный фильтр по metadata (например {"candidate_id": 42})
    """
    query_vec = embed_text(query_text)
    index = get_index()

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )
    return res