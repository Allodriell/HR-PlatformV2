CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS candidate (
    candidate_id BIGSERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL DEFAULT '',
    phone TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS candidate_tag (
    candidate_id BIGINT NOT NULL REFERENCES candidate(candidate_id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (candidate_id, tag)
);

CREATE TABLE IF NOT EXISTS resume (
    resume_id BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES candidate(candidate_id) ON DELETE CASCADE,
    raw_text TEXT NOT NULL,
    source_filename TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS resume_chunk (
    chunk_id BIGSERIAL PRIMARY KEY,
    resume_id BIGINT NOT NULL REFERENCES resume(resume_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    vector_store TEXT NOT NULL DEFAULT 'pinecone',
    vector_id TEXT NOT NULL UNIQUE,
    embed_model TEXT NOT NULL DEFAULT 'text-embedding-3-large',
    embed_dim INTEGER NOT NULL DEFAULT 3072,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (resume_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_resume_candidate_id
    ON resume(candidate_id);

CREATE INDEX IF NOT EXISTS idx_resume_chunk_resume_id
    ON resume_chunk(resume_id);

CREATE TABLE IF NOT EXISTS chat_session (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    current_candidate_id BIGINT REFERENCES candidate(candidate_id) ON DELETE SET NULL,
    current_search_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_message (
    message_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_session(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_id_created_at
    ON chat_message(session_id, created_at);

CREATE TABLE IF NOT EXISTS candidate_search (
    search_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_session(session_id) ON DELETE SET NULL,
    raw_query TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    request_type TEXT NOT NULL DEFAULT 'single_role',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_chat_session_current_search'
    ) THEN
        ALTER TABLE chat_session
            ADD CONSTRAINT fk_chat_session_current_search
            FOREIGN KEY (current_search_id)
            REFERENCES candidate_search(search_id)
            ON DELETE SET NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS candidate_search_result (
    search_id UUID NOT NULL REFERENCES candidate_search(search_id) ON DELETE CASCADE,
    candidate_id BIGINT NOT NULL REFERENCES candidate(candidate_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (search_id, candidate_id),
    UNIQUE (search_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_candidate_search_result_candidate_id
    ON candidate_search_result(candidate_id);
