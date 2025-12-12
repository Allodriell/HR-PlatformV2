import os
from pinecone import Pinecone

api_key = os.environ["PINECONE_API_KEY"]
index_name = os.environ.get("PINECONE_INDEX_NAME", "hr-resume-chunks")

pc = Pinecone(api_key=api_key)

print("=== Список индексов ===")
indexes = pc.list_indexes()
for idx in indexes:
    print(f"- {idx['name']} (dimension={idx['dimension']}, metric={idx['metric']})")

# Проверяем, что нужный индекс есть
names = [i["name"] for i in indexes]
if index_name not in names:
    print(f"\nИндекс {index_name} НЕ найден! Проверь имя индекса.")
    raise SystemExit(0)

print(f"\nИндекс {index_name} найден. Читаю статистику...")

index = pc.Index(index_name)

stats = index.describe_index_stats()
print("Общая статистика по индексу:")
print(f"- всего векторов: {stats['total_vector_count']}")
print(f"- namespaces: {list(stats['namespaces'].keys())}")

# --- Показать несколько векторных ID (по нашему шаблону) ---

# Мы в ingest-скрипте задавали ID векторов как f\"cand-{candidate_id}-chunk-{chunk_index}\"
# Попробуем выборочно их вытащить
sample_ids = [
    f"cand-{cand_id}-chunk-{chunk_idx}"
    for cand_id in range(1, 6)      # кандидаты 1..5
    for chunk_idx in range(0, 5)    # первые 5 чанков
]

fetched = index.fetch(ids=sample_ids)

print("\nНайденные вектора (ID → metadata):")
if not fetched.vectors:
    print("Ничего не найдено по указанным ID (возможно, другие candidate_id / chunk_index).")
else:
    for vid, v in fetched.vectors.items():
        print(f"- {vid}: {v.metadata}")