from rag_engine import build_index, search


model, index, metadata = build_index()

query = "Which shipping modes are used?"

results = search(
    query,
    model,
    index,
    metadata,
    top_k=3
)

for result in results:

    print("\nSOURCE:", result["source"])
    print("DISTANCE:", result["distance"])
    print("TEXT:")
    print(result["text"])