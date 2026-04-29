# Database

Postgres is the source of truth for candidates, resumes, chat state, and search history.
Pinecone is a derived vector index and can be rebuilt from Postgres.

Apply migrations:

```bash
cd backend
source .venv/bin/activate
python scripts/apply_migrations.py
```

The command uses `HR_DB_DSN` from `backend/.env`.
