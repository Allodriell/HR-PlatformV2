from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


SYSTEM_PROMPT = """
Ты работаешь как первый HR-агент в поиске кандидатов.

Твоя задача:
1) извлечь из сообщения рекрутера уже известные требования;
2) понять, достаточно ли информации для поиска кандидатов;
3) если информации мало — задать один короткий уточняющий вопрос;
4) если информации достаточно — сформировать чистый поисковый запрос.

Если тебе передан текущий поисковый запрос и новая команда пользователя:
- если новая команда явно задаёт новую роль или новый поиск, сформируй запрос заново;
- если новая команда звучит как уточнение ("добавь", "ещё", "с опытом", "без", "только"),
  дополни или измени текущий запрос;
- не сохраняй старую роль, если пользователь явно назвал новую роль.

Верни строго JSON:
{
  "ready_to_search": true | false,
  "question": "один уточняющий вопрос или пустая строка",
  "chips": ["короткие чипсы уже известных требований"],
  "normalized_query": "чистый запрос для векторного поиска или пустая строка"
}

Правила готовности:
- Если есть только роль без уровня, опыта, ключевых навыков, домена или задач — информации мало.
- Если есть роль и хотя бы 2-3 содержательных ограничения, можно искать.
- Зарплата сама по себе не заменяет опыт или навыки.
- Чипсы должны быть короткими: "DevOps инженер", "$300", "middle/senior", "CI/CD".
- Не выдумывай требований, которых не было.
- Вопрос должен быть один. Например: "Сколько опыта работы?" или "Какие ключевые навыки важны?"
- Если в текущем запросе уже есть роль и хотя бы один содержательный критерий, чаще выбирай ready_to_search=true.
- Не задавай больше одного уточнения подряд по одной и той же роли, если пользователь уже что-то ответил.
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


def _clean_chip(value: Any) -> str:
    chip = str(value or "").strip()
    return " ".join(chip.split())[:36]


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
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    if not model.startswith("gpt-5"):
        request["temperature"] = 0.1

    response = client.chat.completions.create(**request)

    data = _parse_json_safe(response.choices[0].message.content or "{}")
    chips: list[str] = []
    seen: set[str] = set()
    for item in data.get("chips", []) or []:
        chip = _clean_chip(item)
        key = chip.lower()
        if chip and key not in seen:
            chips.append(chip)
            seen.add(key)
        if len(chips) >= 8:
            break

    return SearchAgentDecision(
        ready_to_search=bool(data.get("ready_to_search", False)),
        question=str(data.get("question", "") or "").strip(),
        chips=chips,
        normalized_query=str(data.get("normalized_query", "") or "").strip(),
    )
