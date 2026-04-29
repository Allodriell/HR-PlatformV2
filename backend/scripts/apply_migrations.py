from __future__ import annotations

from pathlib import Path

import psycopg2
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "db" / "migrations"


def main() -> None:
    load_dotenv(BACKEND_DIR / ".env")

    import os

    dsn = os.environ["HR_DB_DSN"]
    migration_paths = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_paths:
        print("No migrations found.")
        return

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            for path in migration_paths:
                print(f"Applying {path.name}...")
                cur.execute(path.read_text(encoding="utf-8"))

    print("Migrations applied.")


if __name__ == "__main__":
    main()
