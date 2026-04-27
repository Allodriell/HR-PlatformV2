import os
from openai import OpenAI

# Клиент берёт ключ из переменной OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # размерность text-embedding-3-small

def embed_text(text: str) -> list[float]:
    """
    Возвращает эмбеддинг одного текста как список float.
    """
    # OpenAI очень не любит слишком длинные строки — обрежем на всякий случай
    text = text.strip()
    if not text:
        raise ValueError("Пустой текст нельзя заэмбеддить")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    # Берём первый (и единственный) вектор
    return response.data[0].embedding