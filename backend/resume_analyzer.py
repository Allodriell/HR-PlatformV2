from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


SYSTEM_PROMPT = """
Ты анализируешь резюме кандидата для HR-платформы.

Верни строго валидный JSON:
{
  "role": "краткая основная роль кандидата",
  "tags": ["3-6 коротких тегов навыков или технологий"]
}

Правила:
- role должна быть короткой: например "DevOps Engineer", "UX/UI Designer", "Backend Developer".
- tags должны быть только профессиональными навыками, технологиями, инструментами, методологиями или стандартами.
- Хорошие tags: "Figma", "CI/CD", "Python", "Kubernetes", "Human-Centered Design", "Material Design".
- Плохие tags: названия компаний, места работы, имена продуктов работодателя, должности, seniority, годы опыта, личные качества, хобби, "стартап", "команда".
- Не выдумывай навыки, которых нет в резюме.
- Не добавляй больше 6 тегов.
- Не возвращай пояснения вне JSON.
"""


@dataclass
class ResumeAnalysis:
    role: str
    tags: list[str]


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


def _clean_tag(value: Any) -> str:
    tag = str(value or "").strip()
    return " ".join(tag.split())[:40]


def analyze_resume(raw_resume_text: str) -> ResumeAnalysis:
    text = (raw_resume_text or "").strip()
    if not text:
        return ResumeAnalysis(role="", tags=[])

    model = os.environ.get("RESUME_ANALYZER_MODEL", "gpt-5-mini")
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    if not model.startswith("gpt-5"):
        request["temperature"] = 0.1

    response = client.chat.completions.create(**request)

    data = _parse_json_safe(response.choices[0].message.content or "{}")
    role = str(data.get("role", "") or "").strip()

    tags: list[str] = []
    seen: set[str] = set()
    for item in data.get("tags", []) or []:
        tag = _clean_tag(item)
        key = tag.lower()
        if tag and key not in seen:
            tags.append(tag)
            seen.add(key)
        if len(tags) >= 6:
            break

    return ResumeAnalysis(role=role, tags=tags)
