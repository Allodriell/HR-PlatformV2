from vector_pipeline_common import (
    chunk_text,
    get_db_connection,
    upsert_resume_chunks_to_pinecone,
)
from resume_analyzer import analyze_resume


def create_candidate_and_resume(
    full_name: str,
    email: str,
    phone: str,
    role: str,
    raw_resume_text: str,
):
    """
    Создает кандидата и его резюме в БД, затем индексирует резюме в Pinecone.
    """
    analysis = analyze_resume(raw_resume_text)
    resolved_role = role.strip() or analysis.role

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidate (full_name, email, phone, role)
                VALUES (%s, %s, %s, %s)
                RETURNING candidate_id;
                """,
                (full_name, email, phone, resolved_role),
            )
            candidate_id = cur.fetchone()[0]

            for tag in analysis.tags:
                cur.execute(
                    """
                    INSERT INTO candidate_tag (candidate_id, tag)
                    VALUES (%s, %s)
                    ON CONFLICT (candidate_id, tag) DO NOTHING;
                    """,
                    (candidate_id, tag),
                )

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

        chunks = chunk_text(raw_resume_text, target_chars=800, overlap_chars=200)
        print(f"Сформировано чанков: {len(chunks)}")

        upsert_resume_chunks_to_pinecone(resume_id, candidate_id, chunks)
        print(f"Кандидат {candidate_id}, резюме {resume_id} успешно проиндексированы.")

        return {
            "candidate_id": candidate_id,
            "resume_id": resume_id,
            "chunks_count": len(chunks),
            "role": resolved_role,
            "tags": analysis.tags,
        }
    finally:
        conn.close()


def main():
    print("=== Интерактивная загрузка кандидата и резюме ===")

    full_name = input("ФИО кандидата: ").strip()
    email = input("Email кандидата (можно пусто): ").strip()
    phone = input("Телефон кандидата (можно пусто): ").strip()
    role = input("Текущая роль/позиция: ").strip()

    print("Введите резюме. Когда закончите, введите строку: END")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    raw_resume_text = "\n".join(lines).strip()

    if not raw_resume_text:
        print("Пустое резюме, отмена.")
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
