ALTER TABLE resume_chunk
    ALTER COLUMN embed_model SET DEFAULT 'text-embedding-3-large',
    ALTER COLUMN embed_dim SET DEFAULT 3072;
