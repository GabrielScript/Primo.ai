import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path='primo_memory/chroma_db')

# Usa o mesmo modelo de embedding do app
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_collection('primo_knowledge', embedding_function=embedding_fn)

# Simula a busca que o app faz
query = "Quais são os 25 livros que o Primo indica para 2025?"
results = collection.query(
    query_texts=[query],
    n_results=15,
    include=['documents', 'metadatas']
)

print(f"Query: {query}")
print(f"\nTotal de resultados: {len(results['ids'][0])}")

# Conta chunks por vídeo
video_count = {}
for meta in results['metadatas'][0]:
    vid = meta.get('video_id', 'unknown')
    video_count[vid] = video_count.get(vid, 0) + 1

print(f"\nChunks por vídeo:")
for vid, count in sorted(video_count.items(), key=lambda x: -x[1]):
    print(f"  {vid}: {count} chunks")

# Verifica se o vídeo dos 25 livros aparece
target_vid = "TF_OCf8Mh6k"
if target_vid in video_count:
    print(f"\n✅ Vídeo '25 livros' encontrado com {video_count[target_vid]} chunks!")
    
    # Testa a expansão - busca todos os chunks desse vídeo
    all_chunks = collection.get(
        where={"video_id": target_vid},
        include=['documents', 'metadatas']
    )
    print(f"   Total de chunks desse vídeo no banco: {len(all_chunks['ids'])}")
else:
    print(f"\n❌ Vídeo '25 livros' NÃO apareceu na busca!")
    print("   Isso significa que a busca semântica não está conectando a query ao vídeo correto.")
