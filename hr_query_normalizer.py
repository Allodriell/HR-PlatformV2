import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# Инициализация клиента OpenAI (берёт ключ из переменной окружения)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


SYSTEM_PROMPT = """
Ты работаешь внутри HR-платформы как модуль анализа запросов рекрутеров.

Твоя задача:
1) Определить, относится ли текст к поиску кандидата и имеет ли он смысл для HR-поиска.
2) Если да — аккуратно нормализовать запрос: убрать воду, сохранить все важные детали
   и переформулировать запрос одним абзацем.
3) Вернуть результат строго в формате JSON.

Нельзя придумывать новые технологии, навыки или требования, которых нет в исходном запросе.
Нельзя удалять конкретные технологии, домены, уровни опыта, которые явно упомянуты.

Если запрос не про поиск кандидата (шутка, бессмысленный текст, вопрос о платформе) —
отметь это как is_hr_relevant = false.

Структура JSON-ответа:

{
  "intent": "candidate_search" | "platform_question" | "smalltalk" | "nonsense",
  "is_hr_relevant": true | false,
  "confidence": 0.0-1.0,
  "normalized_query": "строка",
  "short_explanation": "строка",
  "request_type": "single_role" | "multiple_roles" | "unclear"
}

Обязательные правила:
- Если is_hr_relevant = false, normalized_query должен быть пустой строкой "".
- Если is_hr_relevant = true, normalized_query должен содержать аккуратно сформулированный запрос на поиск кандидата.
- Ответ ДОЛЖЕН быть строго валидным JSON без пояснений до или после.
"""


@dataclass
class HRQueryNormalizationResult:
    intent: str
    is_hr_relevant: bool
    confidence: float
    normalized_query: str
    short_explanation: str
    request_type: str

    @property
    def should_search_candidates(self) -> bool:
        """
        Мягкий фильтр:
        - не ищем, если запрос не HR-релевантен;
        - не ищем, если intent не candidate_search;
        - не ищем, если уверенность ниже порога.
        """
        if not self.is_hr_relevant:
            return False
        if self.intent != "candidate_search":
            return False
        if self.confidence < 0.4:
            return False
        return True


def _parse_json_safe(text: str) -> dict[str, Any]:
    """
    Аккуратно разбирает JSON-строку.
    Если модель вернула текст вокруг JSON, вырезает первый блок {...}.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)
        raise


def normalize_hr_query(raw_query: str) -> HRQueryNormalizationResult:
    """
    Нормализует HR-запрос и классифицирует его.
    Использует OpenAI API (chat.completions) и возвращает dataclass с результатом.
    """
    text = (raw_query or "").strip()
    if not text:
        return HRQueryNormalizationResult(
            intent="nonsense",
            is_hr_relevant=False,
            confidence=0.0,
            normalized_query="",
            short_explanation="Пустой запрос, поиск кандидатов выполнить нельзя.",
            request_type="unclear",
        )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )

    # В openai 2.x контент лежит здесь:
    content_text: str = response.choices[0].message.content

    data = _parse_json_safe(content_text)

    return HRQueryNormalizationResult(
        intent=str(data.get("intent", "nonsense")),
        is_hr_relevant=bool(data.get("is_hr_relevant", False)),
        confidence=float(data.get("confidence", 0.0)),
        normalized_query=str(data.get("normalized_query", "") or ""),
        short_explanation=str(data.get("short_explanation", "") or ""),
        request_type=str(data.get("request_type", "unclear")),
    )


if __name__ == "__main__":
    print("Тест нормализации HR-запросов. Введите запрос (Ctrl+C для выхода).")
    while True:
        try:
            q = input("\nHR-запрос: ")
        except KeyboardInterrupt:
            print("\nВыход.")
            break

        try:
            res = normalize_hr_query(q)
        except Exception as e:
            print("Ошибка при нормализации:", repr(e))
            continue

        print("\nРезультат нормализации:")
        print(res)
        print("should_search_candidates =", res.should_search_candidates)