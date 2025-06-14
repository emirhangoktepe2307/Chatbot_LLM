import json
import chromadb
from chromadb.utils import embedding_functions

# ChromaDB istemcisini oluştur
client = chromadb.Client()

# SentenceTransformer embedding fonksiyonunu kullan
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()

# Koleksiyonu oluştur
collection = client.create_collection(
    name="ai_egitim_sorulari",
    embedding_function=embedding_function
)

# JSON dosyasından verileri yükle
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Verileri koleksiyona ekle
for i in range(len(data["Sorular"])):
    collection.add(
        documents=[data["Sorular"][i]],
        metadatas=[{
            'title': data["Soru Başlığı"][i],
            'keywords': ', '.join(data["Keywords"][i])  # Listeyi string'e çevir
        }],
        ids=[data["Ticket ID"][i]]
    )

# Test sorgusu
results = collection.query(
    query_texts=["Yapay zeka nedir?"],
    n_results=3
)

print("Test sonuçları:")
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nSonuç {i+1}:")
    print(f"Soru: {doc}")
    print(f"Başlık: {metadata['title']}")
    print(f"Anahtar Kelimeler: {metadata['keywords']}") 