# serch_candidates.py
#
# HR-поисковик кандидатов:
#  - нормализует запрос HR через LLM,
#  - ищет по векторному индексу (Pinecone),
#  - агрегирует результаты по кандидатам,
#  - передаёт управление в candidate_qa.run_candidate_qa()
#    для детального диалога по одному кандидату.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg2
from openai import OpenAI

from hr_query_normalizer import normalize_hr_query
from candidate_qa import run_candidate_qa
from interaction_observer import log_event
from vector_pipeline_common import get_db_connection, get_embedding, index as pinecone_index


# ---------- Модели данных ----------


@dataclass
class VectorMatch:
    candidate_id: int
    resume_id: int
    chunk_index: int
    score: float


@dataclass
class CandidateAggregate:
    candidate_id: int
    total_score: float
    best_score: float
    matches_count: int


# ---------- Инициализация OpenAI ----------

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


NAVIGATION_SYSTEM_PROMPT = """
Ты работаешь как "роутер" внутри HR-платформы.

Тебе даётся:
1) свободная команда пользователя (HR) после того, как он увидел список кандидатов;
2) краткое описание списка кандидатов.

Твоя задача — понять намерение пользователя и вернуть JSON с полями:

{
  "action": "select_candidate" | "refine_search" | "new_search" | "exit" | "show_help",
  "candidate_rank": 1 или 2 или null,
  "candidate_name": "часть ФИО" или "",
  "refinement_text": "что добавить/изменить в поисковом запросе, если action = 'refine_search'"
}

Правила:

- Если пользователь говорит что-то вроде "кандидат 1", "покажи второго", "давай Иванова",
  то action = "select_candidate". В этом случае:
  - candidate_rank ставь номер в списке, если он явно указан;
  - candidate_name используй, если пользователь указал ФИО или его часть.
- Если пользователь задаёт вопрос про конкретного кандидата с упоминанием его имени
  ("что ещё умеет Алексей?", "где до этого работал Соколов?"),
  тоже используй action = "select_candidate" и candidate_name = часть ФИО.
- Если пользователь просит уточнить требования ("добавь английский B2", "только удалённо"),
  то action = "refine_search" и заполни поле refinement_text человеческой фразой.
- Если он говорит "забудь", "начнём сначала", "новый поиск" — action = "new_search".
- Если он говорит "выход", "закончим" — action = "exit".
- Если неясно, что он хочет, поставь action = "show_help".
"""


# ---------- Утилиты ----------


def _parse_json_safe(text: str) -> Dict[str, Any]:
    """
    Пытается распарсить JSON из текста. Если не удаётся, возвращает заглушку.
    """
    text = text.strip()
    if text.startswith("```"):
        # убираем обёртку ```json ... ```
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}


def interpret_navigation_command(user_text: str, candidates_summary: str) -> Dict[str, Any]:
    """
    LLM-роутер: определяет, что пользователь хочет сделать после выдачи кандидатов.
    """
    user_content = (
        "Список кандидатов:\n"
        f"{candidates_summary}\n\n"
        "Команда пользователя:\n"
        f"{user_text}\n\n"
        "Напомню: надо вернуть только JSON со структурой, описанной в инструкции."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": NAVIGATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    data = _parse_json_safe(raw)

    return {
        "action": data.get("action", "show_help"),
        "candidate_rank": data.get("candidate_rank"),
        "candidate_name": data.get("candidate_name", "") or "",
        "refinement_text": data.get("refinement_text", "") or "",
    }


def heuristic_navigation(
    user_text: str,
    aggregated: List[CandidateAggregate],
    info: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Простейшая эвристика, чтобы команды вида
    "где до этого работал Алексей?" или "кандидат 1"
    всегда корректно распознавались как выбор кандидата,
    даже если LLM вернула show_help.
    """
    low = user_text.lower()

    # Новый поиск / выход — дублируем верхнеуровневую логику
    if any(word in low for word in ("забуд", "сначала", "новый поиск")):
        return {"action": "new_search"}
    if any(word in low for word in ("выход", "законч")):
        return {"action": "exit"}

    # "кандидат 1", "кандидата 2" и т.п.
    m = re.search(r"кандидат[а]?\s*(\d+)", low)
    if m:
        try:
            rank = int(m.group(1))
            return {
                "action": "select_candidate",
                "candidate_rank": rank,
                "candidate_name": "",
            }
        except ValueError:
            pass

    # Поиск по имени/фамилии кандидата в пределах текущего списка
    for agg in aggregated:
        meta = info.get(agg.candidate_id, {})
        full_name = str(meta.get("full_name", "")).strip()
        if not full_name:
            continue

        parts = [p for p in full_name.lower().split() if p]
        if not parts:
            continue

        # считаем, что упоминание имени или фамилии достаточно
        if any(p in low for p in parts):
            return {
                "action": "select_candidate",
                "candidate_rank": None,
                "candidate_name": full_name,
            }

    return None


def search_vectors(normalized_query: str, top_k: int = 20) -> List[VectorMatch]:
    """
    Выполняет поиск в Pinecone и возвращает список VectorMatch.
    """
    query_embedding = get_embedding(normalized_query)

    res = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    matches: List[VectorMatch] = []

    for m in res.matches or []:
        md = m.metadata or {}
        try:
            candidate_id = int(md.get("candidate_id"))
            resume_id = int(md.get("resume_id"))
            chunk_index = int(md.get("chunk_index", 0))
        except (TypeError, ValueError):
            continue

        matches.append(
            VectorMatch(
                candidate_id=candidate_id,
                resume_id=resume_id,
                chunk_index=chunk_index,
                score=float(m.score),
            )
        )

    return matches


def aggregate_matches_by_candidate(matches: List[VectorMatch]) -> List[CandidateAggregate]:
    """
    Агрегирует результаты поиска по candidate_id.
    """
    by_candidate: Dict[int, CandidateAggregate] = {}

    for m in matches:
        agg = by_candidate.get(m.candidate_id)
        if agg is None:
            by_candidate[m.candidate_id] = CandidateAggregate(
                candidate_id=m.candidate_id,
                total_score=m.score,
                best_score=m.score,
                matches_count=1,
            )
        else:
            agg.total_score += m.score
            if m.score > agg.best_score:
                agg.best_score = m.score
            agg.matches_count += 1

    aggregated = list(by_candidate.values())
    aggregated.sort(key=lambda a: a.total_score, reverse=True)
    return aggregated


def fetch_candidates_info(candidate_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Загружает из БД базовую информацию по кандидату: ФИО, роль, email.
    """
    if not candidate_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(candidate_ids))
    sql = f"""
        SELECT candidate_id, full_name, role, email
        FROM candidate
        WHERE candidate_id IN ({placeholders})
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, candidate_ids)
            rows = cur.fetchall()
    finally:
        conn.close()

    result: Dict[int, Dict[str, Any]] = {}
    for cid, full_name, role, email in rows:
        result[int(cid)] = {
            "candidate_id": int(cid),
            "full_name": full_name,
            "role": role,
            "email": email,
        }
    return result


def build_candidates_summary(
    aggregated: List[CandidateAggregate],
    info: Dict[int, Dict[str, Any]],
) -> str:
    """
    Строит текстовое описание списка кандидатов для показа HR и для LLM-роутера.
    """
    lines: List[str] = []
    for idx, agg in enumerate(aggregated, start=1):
        meta = info.get(agg.candidate_id, {})
        full_name = meta.get("full_name", f"Кандидат #{agg.candidate_id}")
        role = meta.get("role", "<роль не указана>")
        email = meta.get("email", "")

        lines.append(
            f"{idx}. {full_name} — {role} (id={agg.candidate_id})\n"
            f"   Итоговый score: {agg.total_score:.4f}\n"
            f"   E-mail: {email}"
        )

    return "\n".join(lines)


# ---------- Основной сценарий поиска ----------


def search_candidates_interactive() -> None:
    """
    Главный интерактивный сценарий поиска кандидатов.
    """
    print("=== Режим поиска кандидатов ===")
    print("HR-поиск кандидатов. Нажмите Ctrl+C для выхода в любой момент.\n")

    current_base_query: Optional[str] = None

    while True:
        try:
            if current_base_query is None:
                raw_query = input(
                    'Опишите, какого кандидата вы ищете (или "выход" для завершения): '
                ).strip()
                if not raw_query:
                    continue

                if raw_query.lower() in {"выход", "exit", "quit"}:
                    print("Выход из режима поиска.")
                    return

                current_base_query = raw_query
            else:
                raw_query = current_base_query

            # --- нормализация запроса HR ---
            log_event(
                "search_raw_query",
                {"raw_query": raw_query},
            )

            norm = normalize_hr_query(raw_query)

            log_event(
                "search_normalized_query",
                {
                    "raw_query": raw_query,
                    "normalized_query": getattr(norm, "normalized_query", ""),
                    "is_hr_relevant": getattr(norm, "is_hr_relevant", True),
                    "confidence": getattr(norm, "confidence", None),
                    "short_explanation": getattr(norm, "short_explanation", ""),
                    "request_type": getattr(norm, "request_type", ""),
                    "intent": getattr(norm, "intent", ""),
                    "should_search_candidates": getattr(
                        norm, "should_search_candidates", True
                    ),
                },
            )

            if not getattr(norm, "is_hr_relevant", True) or not getattr(
                norm, "should_search_candidates", True
            ):
                print("\nКажется, запрос не относится к поиску кандидатов.")
                explanation = getattr(norm, "short_explanation", "")
                if explanation:
                    print(explanation)
                print("Попробуйте сформулировать запрос иначе.\n")
                current_base_query = None
                continue

            normalized_query = getattr(norm, "normalized_query", raw_query)
            print("\nНормализованный запрос:")
            print(normalized_query)
            print()

            # --- поиск в векторном индексе ---
            matches = search_vectors(normalized_query, top_k=20)
            aggregated = aggregate_matches_by_candidate(matches)

            candidate_ids = [a.candidate_id for a in aggregated]
            info = fetch_candidates_info(candidate_ids)

            log_event(
                "search_results",
                {
                    "normalized_query": normalized_query,
                    "candidate_ids": candidate_ids,
                    "scores": [a.total_score for a in aggregated],
                },
            )

            if not aggregated:
                print("Кандидаты не найдены. Попробуйте изменить запрос.\n")
                current_base_query = None
                continue

            # --- выводим список кандидатов ---
            print("Найденные кандидаты:")
            summary = build_candidates_summary(aggregated, info)
            print(summary)
            print()

            # --- навигация после выдачи ---
            while True:
                print("Что вы хотите сделать дальше?")
                print("Можете писать в свободной форме, например:")
                print('  "покажи кандидата 2",')
                print('  "давай Иванова",')
                print('  "добавь требование английского",')
                print('  "забудь, начнём сначала",')
                print('  "выход".\n')

                nav_text = input("Ваше сообщение: ").strip()
                if not nav_text:
                    continue

                low = nav_text.lower()
                if low in {"выход", "exit", "quit", "закончим", "стоп"}:
                    print("Выход из режима поиска.")
                    return

                # сначала пробуем эвристику
                heur = heuristic_navigation(nav_text, aggregated, info)
                if heur is not None:
                    nav = {
                        "action": heur.get("action", "show_help"),
                        "candidate_rank": heur.get("candidate_rank"),
                        "candidate_name": heur.get("candidate_name", "") or "",
                        "refinement_text": heur.get("refinement_text", "") or "",
                    }
                else:
                    # затем — LLM-роутер
                    nav = interpret_navigation_command(nav_text, summary)

                log_event(
                    "navigation_raw_command",
                    {
                        "user_text": nav_text,
                        "candidates_summary": summary,
                    },
                )
                log_event(
                    "navigation_decision",
                    {
                        "user_text": nav_text,
                        "action": nav.get("action"),
                        "candidate_rank": nav.get("candidate_rank"),
                        "candidate_name": nav.get("candidate_name"),
                        "refinement_text": nav.get("refinement_text"),
                    },
                )

                action = nav.get("action", "show_help")

                if action == "exit":
                    print("Выход из режима поиска.")
                    return

                if action == "new_search":
                    current_base_query = None
                    break  # к началу главного цикла

                if action == "refine_search":
                    refinement = nav.get("refinement_text", "").strip()
                    if refinement:
                        print("\nУточняем запрос с учётом ваших пожеланий:")
                        print(refinement)
                        current_base_query = normalized_query + ". " + refinement
                    else:
                        print("Не удалось понять уточнение. Попробуйте сформулировать иначе.")
                    break  # новый поиск

                if action == "select_candidate":
                    rank = nav.get("candidate_rank", None)
                    cand_name = (nav.get("candidate_name") or "").strip().lower()

                    selected_id: Optional[int] = None

                    # 1) пробуем номер
                    if isinstance(rank, int) and 1 <= rank <= len(aggregated):
                        selected_id = aggregated[rank - 1].candidate_id

                    # 2) пробуем поиск по имени в пределах текущего списка
                    if selected_id is None and cand_name:
                        for a in aggregated:
                            meta = info.get(a.candidate_id, {})
                            full_name = str(meta.get("full_name", "")).lower()
                            if cand_name in full_name:
                                selected_id = a.candidate_id
                                break

                    if selected_id is None:
                        print(
                            "Не удалось однозначно определить кандидата по этому запросу."
                        )
                        print(
                            'Попробуйте что-то вроде: "кандидат 1" или точное ФИО.\n'
                        )
                        continue

                    # --- запускаем отдельный режим Q&A по кандидату ---
                    mode_result = run_candidate_qa(
                        selected_id,
                        initial_question=nav_text,
                    )

                    if mode_result == "exit":
                        return
                    if mode_result == "new_search":
                        current_base_query = None
                        break  # к началу главного цикла
                    # "same_search" — продолжаем с тем же списком кандидатов
                    continue

                # если action не распознан
                print(
                    "Я не до конца понял команду. Попробуйте указать номер кандидата или уточнение к запросу.\n"
                )

        except KeyboardInterrupt:
            print("\nЗавершение режима поиска по Ctrl+C.")
            return
        except psycopg2.Error as e:
            print("\nОшибка работы с БД:", e)
            return
        except Exception as e:
            print("\nНеожиданная ошибка:", e)
            return


def main() -> None:
    search_candidates_interactive()


if __name__ == "__main__":
    main()