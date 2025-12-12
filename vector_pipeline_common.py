import os
import math
from typing import List, Tuple

import psycopg2
from pinecone import Pinecone
from openai import OpenAI


# === Инициализация клиентов ===

HR_DB_DSN = os.environ["HR_DB_DSN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# === Эмбеддинги ===

def get_embedding(text: str) -> List[float]:
    """
    Получает эмбеддинг текста через OpenAI text-embedding-3-small.
    """
    text = text.strip()
    if not text:
        # На всякий случай возвращаем нулевой вектор
        return [0.0] * 1536

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# === Чанкование текста ===

def split_into_sentences(text: str) -> List[str]:
    """
    Очень простое разбиение на "предложения" по точкам/переносам.
    Без внешних библиотек.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = []
    buf = []

    for ch in text:
        buf.append(ch)
        if ch in [".", "!", "?", "\n"]:
            sentence = "".join(buf).strip()
            if sentence:
                parts.append(sentence)
            buf = []

    # хвост, если не закончился знаком
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    return parts


def chunk_text(
    text: str,
    target_chars: int = 800,
    overlap_chars: int = 200
) -> List[str]:
    """
    Чанкование резюме с перекрытием по символам.
    1. Разбиваем на "предложения".
    2. Набираем чанки примерно по target_chars.
    3. Перекрываем чанки на overlap_chars символов.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Сначала собираем "сырые" чанки (без overlap)
    raw_chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        if len(current) + 1 + len(sent) <= target_chars:
            current += " " + sent
        else:
            raw_chunks.append(current)
            current = sent

    if current:
        raw_chunks.append(current)

    if len(raw_chunks) == 1:
        return raw_chunks

    # Теперь добавим overlap по символам между соседними чанками
    final_chunks: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            prev = raw_chunks[i - 1]
            # берём хвост предыдущего чанка
            tail = prev[-overlap_chars:]
            merged = tail + " " + chunk
            final_chunks.append(merged)

    return final_chunks


# === Подключение к БД ===

def get_db_connection():
    return psycopg2.connect(HR_DB_DSN)


# === Работа с Pinecone ===

def upsert_resume_chunks_to_pinecone(
    resume_id: int,
    candidate_id: int,
    chunks: List[str]
) -> None:
    """
    Для каждого чанка резюме:
    - считаем эмбеддинг через OpenAI,
    - отправляем вектор в Pinecone,
    - сохраняем сам чанк и метаданные в таблицу resume_chunk.
    """
    if not chunks:
        return

    vectors = []

    # Сначала работаем с БД: вставляем чанки и метаданные
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for idx, chunk_text in enumerate(chunks):
                chunk_text = (chunk_text or "").strip()
                if not chunk_text:
                    continue

                # считаем эмбеддинг
                embedding = get_embedding(chunk_text)
                embed_dim = len(embedding)

                # формируем ID вектора
                vector_id = f"resume-{resume_id}-chunk-{idx}"

                # добавляем в список для Pinecone
                metadata = {
                    "candidate_id": candidate_id,
                    "resume_id": resume_id,
                    "chunk_index": idx,
                    "text_preview": chunk_text[:300],
                }
                vectors.append((vector_id, embedding, metadata))

                # записываем чанк в БД
                cur.execute(
                    """
                    INSERT INTO resume_chunk (
                        resume_id,
                        chunk_index,
                        chunk_text,
                        vector_store,
                        vector_id,
                        embed_model,
                        embed_dim
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        resume_id,
                        idx,
                        chunk_text,
                        "pinecone",                 # vector_store
                        vector_id,
                        "text-embedding-3-small",   # embed_model
                        embed_dim,                  # embed_dim
                    ),
                )

        conn.commit()
    finally:
        conn.close()

    # После успешной записи в БД — отправляем векторы в Pinecone
    if vectors:
        index.upsert(vectors)