import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict


# Путь к лог-файлу можно переопределить через переменную окружения
LOG_PATH = os.environ.get("HR_AUDIT_LOG_PATH", "hr_audit.log")


# Один session_id на запуск приложения
SESSION_ID = str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Универсальный логгер событий платформы.

    Пишет одну строку JSON в файл LOG_PATH:
    {
      "ts": "...",
      "session_id": "...",
      "event_type": "...",
      "data": { ... }
    }

    Если лог не удаётся записать (нет прав, диск и т.п.), платформа продолжает работать.
    """
    record = {
        "ts": _now_iso(),
        "session_id": SESSION_ID,
        "event_type": event_type,
        "data": payload,
    }

    try:
        line = json.dumps(record, ensure_ascii=False)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # В проде лучше логировать ошибку отдельно.
        # Для дипломного прототипа просто не ломаем основной поток.
        pass