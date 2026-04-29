# candidate_qa.py
#
# Скрипт для вопросов по ОДНОМУ кандидату.
# Используется из serch_candidates.py (run_candidate_qa),
# а также может запускаться самостоятельно.

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from vector_pipeline_common import get_db_connection
from interaction_observer import log_event


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


CANDIDATE_ASSISTANT_SYSTEM_PROMPT = """
Ты ассистент внутри HR-платформы.

Тебе даётся резюме одного кандидата.
Твоя задача — отвечать на вопросы HR строго на основе этого резюме, ничего не выдумывая.

Правила:

1. Если в резюме есть нужная информация:
   - процитируй релевантные фрагменты (в виде блоков с >),
   - затем коротко прокомментируй.

2. Если в резюме НЕТ прямого указания того, о чём спрашивает HR:
   - так и скажи, что в резюме этого явно нет,
   - не фантазируй и не достраивай детали.

3. Сохраняй последовательность:
   - учитывай предыдущие ответы и вопросы в текущей сессии,
   - если обнаруживаешь, что ранний ответ можно уточнить, аккуратно это поясни
     (например: "раньше я писал, что компания не указана, однако в другом фрагменте есть упоминание ...").

Верни строго валидный JSON:
{
  "answer": "краткий ответ HR",
  "evidence_quote": "точная короткая цитата из резюме, которая подтверждает ответ"
}

Правила для evidence_quote:
- копируй фрагмент дословно из резюме;
- если прямого подтверждения нет, верни пустую строку "";
- не делай evidence_quote длиннее 500 символов.
"""


# ---------- работа с БД ----------


def _parse_json_safe(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        return {"answer": text, "evidence_quote": ""}

def fetch_candidate_meta(candidate_id: int) -> Optional[Dict[str, Any]]:
    sql = """
        SELECT candidate_id, full_name, role, email, phone
        FROM candidate
        WHERE candidate_id = %s
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (candidate_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    cand_id, full_name, role, email, phone = row
    return {
        "candidate_id": cand_id,
        "full_name": full_name,
        "role": role,
        "email": email,
        "phone": phone,
    }


def fetch_candidate_tags(candidate_id: int) -> List[str]:
    sql = """
        SELECT tag
        FROM candidate_tag
        WHERE candidate_id = %s
        ORDER BY tag
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (candidate_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]


def fetch_candidate_chunks(candidate_id: int) -> List[str]:
    sql = """
        SELECT rc.chunk_text
        FROM resume_chunk rc
        JOIN resume r ON rc.resume_id = r.resume_id
        WHERE r.candidate_id = %s
        ORDER BY rc.resume_id, rc.chunk_index
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (candidate_id,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]


def fetch_candidate_resume_text(candidate_id: int) -> Optional[str]:
    sql = """
        SELECT raw_text
        FROM resume
        WHERE candidate_id = %s
        ORDER BY resume_id DESC
        LIMIT 1
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (candidate_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    return row[0]


def build_candidate_resume_context(candidate_id: int) -> Optional[Dict[str, Any]]:
    """
    Готовит структурированный контекст кандидата для API и LLM-вопросов.
    """
    meta = fetch_candidate_meta(candidate_id)
    if meta is None:
        return None

    chunks = fetch_candidate_chunks(candidate_id)
    if not chunks:
        return None

    resume_text = fetch_candidate_resume_text(candidate_id) or "\n\n".join(chunks)

    return {
        "meta": meta,
        "tags": fetch_candidate_tags(candidate_id),
        "chunks": chunks,
        "resume_text": resume_text,
    }


def answer_candidate_question(
    candidate_id: int,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Возвращает ответ по одному кандидату в JSON-совместимом виде.
    """
    context = build_candidate_resume_context(candidate_id)
    if context is None:
        raise ValueError(f"Кандидат с id={candidate_id} не найден или у него нет резюме.")

    meta = context["meta"]
    resume_text = context["resume_text"]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": CANDIDATE_ASSISTANT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Вот фрагменты резюме кандидата. Используй их как единственный источник фактов:\n\n"
                f"{resume_text}"
            ),
        },
        {
            "role": "assistant",
            "content": "Я изучил резюме кандидата. Готов отвечать на вопросы HR.",
        },
    ]

    for item in history or []:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    log_event(
        "candidate_question",
        {"candidate_id": candidate_id, "question": question},
    )

    model = os.environ.get("CANDIDATE_QA_MODEL", "gpt-5.1")
    request: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if not model.startswith("gpt-5"):
        request["temperature"] = 0.0

    response = client.chat.completions.create(**request)
    content_text = response.choices[0].message.content or ""
    data = _parse_json_safe(content_text)
    answer = str(data.get("answer", "") or content_text)
    evidence_quote = str(data.get("evidence_quote", "") or "")

    log_event(
        "candidate_answer",
        {
            "candidate_id": candidate_id,
            "question": question,
            "answer": answer,
            "evidence_quote": evidence_quote,
        },
    )

    return {
        "candidate": meta,
        "answer": answer,
        "evidence_quote": evidence_quote,
        "history": messages + [{"role": "assistant", "content": answer}],
    }


# ---------- основной цикл Q&A по кандидату ----------

def run_candidate_qa(
    candidate_id: int,
    initial_question: Optional[str] = None,
) -> str:
    """
    Интерактивный режим вопросов по одному кандидату.

    initial_question — первый вопрос, который пользователь уже задал
    (например: "Где до этого работал Алексей?").

    Возвращает:
      "same_search" — вернуться к списку кандидатов (продолжить поиск),
      "new_search"  — начать новый поиск,
      "exit"        — выйти из программы.
    """
    meta = fetch_candidate_meta(candidate_id)
    if meta is None:
        print(f"\nКандидат с id={candidate_id} не найден в БД.")
        return "same_search"

    full_name = meta.get("full_name", f"Кандидат #{candidate_id}")
    role = meta.get("role", "<роль не указана>")

    chunks = fetch_candidate_chunks(candidate_id)
    if not chunks:
        print(f"\nПо кандидату #{candidate_id} нет сохранённых чанков резюме.")
        return "same_search"

    resume_text = fetch_candidate_resume_text(candidate_id) or "\n\n".join(chunks)

    print(f"\nВы смотрите кандидата #{candidate_id}: {full_name} — {role}")
    print(
        "Задавайте вопросы по этому резюме.\n"
        "Примеры:\n"
        "  Где в резюме указано, что он работал в Apple?\n"
        "  Что ещё интересного умеет этот кандидат?\n"
        "Если хотите вернуться к результатам поиска, скажите что-то вроде:\n"
        "  \"вернись к списку\" или \"показать других кандидатов\".\n"
        "Для полного сброса поиска можно сказать \"забудь\" или \"начнём сначала\".\n"
        "Для выхода — \"выход\", \"закончим\" и т.п.\n"
    )

    # --- формируем общий контекст диалога ---
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": CANDIDATE_ASSISTANT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Вот фрагменты резюме кандидата. Используй их как единственный источник фактов:\n\n"
                f"{resume_text}"
            ),
        },
        {
            "role": "assistant",
            "content": "Я изучил резюме кандидата. Готов отвечать на вопросы HR.",
        },
    ]

    def ask_llm(question: str) -> str:
        """Вспомогательная функция: добавить вопрос в историю и получить ответ."""
        messages.append({"role": "user", "content": question})

        model = os.environ.get("CANDIDATE_QA_MODEL", "gpt-5.1")
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if not model.startswith("gpt-5"):
            request["temperature"] = 0.0

        response = client.chat.completions.create(**request)
        answer = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": answer})
        return answer

    # ---------- первый вопрос, если он уже задан ----------
    if initial_question:
        q = initial_question.strip()
        if q:
            low = q.lower()

            if "выход" in low or "законч" in low:
                return "exit"
            if "забуд" in low or "сначала" in low or "новый поиск" in low:
                return "new_search"
            if "список" in low or "других кандидатов" in low or "вернись" in low:
                return "same_search"

            log_event(
                "candidate_question",
                {"candidate_id": candidate_id, "question": q},
            )
            answer = ask_llm(q)
            log_event(
                "candidate_answer",
                {
                    "candidate_id": candidate_id,
                    "question": q,
                    "answer": answer,
                },
            )

            print("\nОтвет ассистента:\n")
            print(answer)
            print("\n---\n")

    # ---------- основной цикл последующих вопросов ----------
    while True:
        try:
            q = input("Ваш запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return "exit"

        if not q:
            continue

        low = q.lower()

        if "выход" in low or "законч" in low:
            return "exit"
        if "забуд" in low or "сначала" in low or "новый поиск" in low:
            return "new_search"
        if "список" in low or "других кандидатов" in low or "вернись" in low:
            return "same_search"

        log_event(
            "candidate_question",
            {"candidate_id": candidate_id, "question": q},
        )
        answer = ask_llm(q)
        log_event(
            "candidate_answer",
            {
                "candidate_id": candidate_id,
                "question": q,
                "answer": answer,
            },
        )

        print("\nОтвет ассистента:\n")
        print(answer)
        print("\n---\n")


# ---------- самостоятельный запуск скрипта (опционально) ----------

def _select_candidate_interactive() -> Optional[int]:
    text = input("Введите id кандидата или часть ФИО: ").strip()
    if not text:
        return None

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if text.isdigit():
                cur.execute(
                    """
                    SELECT candidate_id, full_name, role
                    FROM candidate
                    WHERE candidate_id = %s
                    """,
                    (int(text),),
                )
            else:
                cur.execute(
                    """
                    SELECT candidate_id, full_name, role
                    FROM candidate
                    WHERE full_name ILIKE %s
                    ORDER BY candidate_id
                    """,
                    (f"%{text}%",),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print("Кандидаты не найдены.")
        return None

    if len(rows) == 1:
        cand_id, full_name, role = rows[0]
        print(f"Выбран кандидат #{cand_id}: {full_name} — {role}")
        return cand_id

    print("Найдено несколько кандидатов:")
    for cand_id, full_name, role in rows:
        print(f"  {cand_id}: {full_name} — {role}")

    choice = input("Введите id нужного кандидата: ").strip()
    if not choice.isdigit():
        return None
    return int(choice)


if __name__ == "__main__":
    print("Режим вопросов по кандидату.")
    cid = _select_candidate_interactive()
    if cid is not None:
        run_candidate_qa(cid)
