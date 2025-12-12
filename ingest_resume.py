import sys
from textwrap import dedent

from vector_pipeline_common import (
    get_db_connection,
    chunk_text,
    upsert_resume_chunks_to_pinecone,
)


def create_candidate_and_resume(
    full_name: str,
    email: str,
    phone: str,
    role: str,
    raw_resume_text: str,
):
    """
    Создаёт кандидата и его резюме в БД,
    затем чанкeт текст и отправляет чанки в Pinecone.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # создаём кандидата
            cur.execute(
                """
                INSERT INTO candidate (full_name, email, phone, role)
                VALUES (%s, %s, %s, %s)
                RETURNING candidate_id;
                """,
                (full_name, email, phone, role),
            )
            candidate_id = cur.fetchone()[0]

            # создаём запись резюме
            cur.execute(
                """
                INSERT INTO resume (candidate_id, raw_text)
                VALUES (%s, %s)
                RETURNING resume_id;
                """,
                (candidate_id, raw_resume_text),
            )
            resume_id = cur.fetchone()[0]

        conn.commit()

        # чанкование и отправка в Pinecone
        chunks = chunk_text(raw_resume_text, target_chars=800, overlap_chars=200)
        print(f"Сформировано чанков: {len(chunks)}")

        upsert_resume_chunks_to_pinecone(resume_id, candidate_id, chunks)
        print(f"Кандидат {candidate_id}, резюме {resume_id} успешно проиндексированы.")

    finally:
        conn.close()


def main():
    print("=== Интерактивная загрузка кандидата и резюме ===")

    full_name = input("ФИО кандидата: ").strip()
    email = input("Email кандидата (можно пусто): ").strip()
    phone = input("Телефон кандидата (можно пусто): ").strip()
    role = input("Текущая роль/позиция: ").strip()

    print("Введите резюме. Когда закончите — введите строку: END")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    raw_resume_text = "\n".join(lines).strip()

    if not raw_resume_text:
        print("Пустое резюме — отмена.")
        return

    create_candidate_and_resume(
        full_name=full_name,
        email=email,
        phone=phone,
        role=role,
        raw_resume_text=raw_resume_text,
    )


if __name__ == "__main__":
    main()