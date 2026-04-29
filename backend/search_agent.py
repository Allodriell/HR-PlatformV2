from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


SYSTEM_PROMPT = """
Ты работаешь как редактор HR-запроса внутри платформы поиска кандидатов.

Твоя задача — не собирать теги и не навязывать фиксированную анкету.
Представь, что normalized_query — это отдельное поле состояния в UI.
Каждое сообщение пользователя является командой отредактировать это поле.

Твоя задача:
1) понять реальную потребность рекрутера;
2) очистить пользовательский текст от разговорного и семантического шума;
3) сформировать нормальный поисковый запрос для векторного поиска;
4) задать один короткий уточняющий вопрос только если без ответа запрос остается слишком неоднозначным.

Если тебе передан текущий поисковый запрос и новая команда пользователя:
- если новая команда явно задаёт новую роль или новый поиск, сформируй запрос заново;
- если новая команда звучит как уточнение, дополнение или поправка,
  дополни или измени текущий запрос;
- не сохраняй старую роль, если пользователь явно назвал новую роль.
- не вставляй новую команду пользователя дословно как "дополнительные требования";
  перепиши весь normalized_query как один цельный, аккуратный HR-запрос.

Верни строго JSON:
{
  "ready_to_search": true | false,
  "question": "один уточняющий вопрос или пустая строка",
  "chips": [],
  "normalized_query": "текущий чистый запрос для векторного поиска"
}

Правила:
- normalized_query заполняй всегда, даже если задаешь уточняющий вопрос.
- Не выдумывай требований, которых пользователь не говорил.
- Каждое новое сообщение пользователя должно повлиять на normalized_query:
  добавить новое требование, изменить старое, убрать отмененное требование или явно зафиксировать,
  что критерий не важен.
- Нельзя возвращать normalized_query без изменений, если пользователь написал новое содержательное сообщение.
- normalized_query должен звучать как итоговый запрос рекрутера, а не как история переписки.
- normalized_query — это текст для векторного поиска по резюме. В нем не должно быть
  технических комментариев, мета-пояснений и полей, которых пользователь не указал.
- Не пиши в normalized_query фразы вроде "поиск специалиста", "рекрутинг одного специалиста",
  "дополнительные требования не указаны", "уровень опыта не указан",
  "формат занятости не указан", "локация не указана".
- Не добавляй скобки с отсутствующими параметрами. Если параметр неизвестен,
  просто не упоминай его.
- Сохраняй отрицательные и мягкие ограничения: "опыт не важен", "без React", "можно junior".
- Если пользователь говорит, что опыт, уровень, домен, зарплата или другой параметр не важен,
  не продолжай уточнять этот параметр.
- Если пользователь уже ответил на уточняющий вопрос, чаще выбирай ready_to_search=true.
- Не требуй обязательного опыта, уровня, домена или количества навыков. Это не анкета.
- Задавай вопрос только если непонятна сама роль/направление или запрос слишком общий
  вроде "нужен человек", "кто-нибудь", "специалист".
- Если роль понятна, можно искать даже с коротким запросом.
- chips всегда возвращай пустым массивом [] — UI больше их не использует.

Примеры:
Пользователь: "дизайнер"
Ответ: ready_to_search=false, question="Какой дизайнер нужен?", normalized_query="Найти дизайнера"

Текущий запрос: "Найти дизайнера"
Новая команда: "Мне нужен человек который работает с Motion и UI/UX опыт работы не так важен"
Ответ: ready_to_search=true, question="", normalized_query="Найти дизайнера, который работает с Motion и UI/UX. Опыт работы не является жестким критерием."

Пользователь: "ищу повара, главное итальянская кухня, опыт не принципиален"
Ответ: ready_to_search=true, question="", normalized_query="Найти повара со знанием итальянской кухни. Опыт работы не является жестким критерием."
"""

REWRITE_PROMPT = """
Ты редактор HR-запроса.

Есть текущий поисковый запрос и новая команда пользователя.
Перепиши запрос как один цельный, аккуратный HR-запрос для векторного поиска.

Правила:
- не вставляй команду пользователя дословно;
- не используй формулировки "дополнительные требования", "пользователь сказал", "уточнение";
- не добавляй скобки с отсутствующими параметрами;
- не пиши, что опыт, локация, формат занятости или другие параметры "не указаны";
- сохрани смысл текущего запроса и новой команды;
- если новая команда задает новую профессию или новый поиск, замени старый запрос;
- верни строго JSON: {"normalized_query": "..."}.
"""


@dataclass
class SearchAgentDecision:
    ready_to_search: bool
    question: str
    chips: list[str]
    normalized_query: str


def _parse_json_safe(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _clean_normalized_query(value: str) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return ""

    meta_terms = (
        r"не\s+указа",
        r"уров(?:е|е)нь\s+опыта",
        r"формат\s+занятости",
        r"локац",
        r"дополнительн(?:ые|ых)?\s+(?:требован|услов)",
        r"поиск\s+специалиста",
        r"рекрутинг",
    )
    meta_pattern = "|".join(meta_terms)

    text = re.sub(
        rf"\s*\([^)]*(?:{meta_pattern})[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    noise_patterns = [
        r"\bпоиск\s+специалиста\b",
        r"\bрекрутинг\s+одного\s+специалиста\b",
        r"\bдополнительн(?:ые|ых)?\s+(?:требования|условия|требования/условия)\s+не\s+указан[ыоа]?\b",
        r"\bуров(?:е|е)нь\s+опыта\s+не\s+указан[ао]?\b",
        r"\bформат\s+занятости\s+не\s+указан[ао]?\b",
        r"\bлокац(?:ия|ии|ию)\s+не\s+указан[ао]?\b",
        r"\bне\s+указан[ыоа]?\b",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\(([^)]*)\)", r"\1", text)
    text = re.sub(r"\s+[—-]\s*(?=[,.!?;:]|$)", "", text)
    text = re.sub(r"\s+[—-]\s+", " — ", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){2,}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" \t\r\n-—,.;:")

    return text


def _call_json_model(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if not model.startswith("gpt-5"):
        request["temperature"] = 0.1

    response = client.chat.completions.create(**request)
    return _parse_json_safe(response.choices[0].message.content or "{}")


def _rewrite_prompt_with_update(model: str, current_prompt: str, update: str) -> str:
    current = current_prompt.strip()
    command = update.strip()

    if not current:
        return _clean_normalized_query(command)
    if not command:
        return _clean_normalized_query(current)

    try:
        data = _call_json_model(
            model,
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Текущий поисковый запрос:\n{current}\n\n"
                        f"Новая команда пользователя:\n{command}"
                    ),
                },
            ],
        )
        rewritten = _clean_normalized_query(str(data.get("normalized_query", "") or ""))
        if rewritten and _normalize_text(rewritten) != _normalize_text(current):
            return rewritten
    except Exception:
        pass

    return _clean_normalized_query(f"{current}. {command}")


def decide_search_next_step(message: str, current_prompt: str = "") -> SearchAgentDecision:
    text = (message or "").strip()
    if not text:
        return SearchAgentDecision(
            ready_to_search=False,
            question="Кого ищем?",
            chips=[],
            normalized_query="",
        )

    user_content = text
    if current_prompt.strip():
        user_content = (
            f"Текущий поисковый запрос:\n{current_prompt.strip()}\n\n"
            f"Новая команда пользователя:\n{text}"
        )

    model = os.environ.get("SEARCH_AGENT_MODEL", "gpt-4.1-mini")
    data = _call_json_model(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    normalized_query = _clean_normalized_query(str(data.get("normalized_query", "") or ""))
    ready_to_search = bool(data.get("ready_to_search", False))
    question = str(data.get("question", "") or "").strip()

    if not normalized_query:
        normalized_query = _rewrite_prompt_with_update(model, current_prompt, text)
    elif current_prompt.strip():
        normalized_current = _normalize_text(current_prompt)
        normalized_query_text = _normalize_text(normalized_query)

        model_kept_old_prompt = normalized_query_text == normalized_current

        if model_kept_old_prompt:
            normalized_query = _rewrite_prompt_with_update(model, current_prompt, text)

    normalized_query = _clean_normalized_query(normalized_query)

    return SearchAgentDecision(
        ready_to_search=ready_to_search,
        question=question,
        chips=[],
        normalized_query=normalized_query,
    )
