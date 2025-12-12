import os
from pinecone import Pinecone

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key:
        raise ValueError("Ошибка: переменная окружения PINECONE_API_KEY не установлена.")

    if not index_name:
        raise ValueError("Ошибка: переменная окружения PINECONE_INDEX_NAME не установлена.")

    pc = Pinecone(api_key=api_key)

    print(f"Подключение к индексу: {index_name}")
    index = pc.Index(index_name)

    # Удаляем ВСЕ вектора в индексе
    print("Удаляю все вектора...")
    index.delete(delete_all=True)

    # Проверяем состояние
    stats = index.describe_index_stats()

    print("Индекс очищен!")
    print("Текущее состояние индекса:")
    print(f" - Всего векторов: {stats.get('total_vector_count')}")
    print(f" - Размерность: {stats.get('dimension')}")
    print(f" - Пространства имен: {stats.get('namespaces')}")

if __name__ == "__main__":
    main()