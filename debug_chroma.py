import chromadb

client = chromadb.PersistentClient(path='primo_memory/chroma_db')
collection = client.get_collection('primo_knowledge')

# Busca chunks com video_id específico
results = collection.get(
    where={"video_id": "TF_OCf8Mh6k"},
    include=['metadatas']
)

print(f"Chunks do video '25 livros': {len(results['ids'])}")

# Se não encontrou, verifica alguns metadados para ver como está salvo
if len(results['ids']) == 0:
    print("\nNenhum chunk encontrado com esse video_id!")
    print("Verificando amostra de metadados existentes...")
    
    sample = collection.get(limit=5, include=['metadatas'])
    for i, meta in enumerate(sample['metadatas']):
        print(f"  Chunk {i}: {meta}")
