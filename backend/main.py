"""
Главная точка входа HR-платформы.

Режимы:
1) Загрузка нового резюме и индексация в Pinecone.
2) Поиск кандидатов с нормализацией HR-запроса и LLM-навигацией.
"""

import sys

import ingest_resume
import serch_candidates


def run_ingest_mode() -> None:
    """
    Режим загрузки резюме.
    Делегирует выполнение модулю ingest_resume.
    """
    print("\n=== Режим загрузки резюме ===")
    try:
        # ожидается, что внутри ingest_resume.main() уже есть
        # вся логика диалога (ввод резюме, создание кандидата, индексация)
        ingest_resume.main()
    except KeyboardInterrupt:
        print("\nЗавершение режима загрузки по Ctrl+C.")


def run_search_mode() -> None:
    """
    Режим поиска кандидатов.
    Делегирует выполнение модулю serch_candidates.
    """
    print("\n=== Режим поиска кандидатов ===")
    try:
        serch_candidates.main()
    except KeyboardInterrupt:
        print("\nЗавершение режима поиска по Ctrl+C.")


def main() -> None:
    """
    Главное меню HR-платформы в консоли.
    Даёт выбор между загрузкой резюме и поиском кандидатов.
    """
    print("HR-платформа (консольный прототип)")
    print("Нажмите Ctrl+C в любой момент, чтобы прервать текущую операцию.\n")

    while True:
        print("Выберите режим работы:")
        print("  1 — Загрузить новое резюме")
        print("  2 — Поиск кандидатов")
        print("  0 — Выход")
        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            run_ingest_mode()
            print()
        elif choice == "2":
            run_search_mode()
            print()
        elif choice == "0":
            print("Выход из HR-платформы.")
            return
        else:
            print("Неизвестная команда. Введите 1, 2 или 0.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗавершение работы по Ctrl+C.")
        sys.exit(0)