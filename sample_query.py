import weaviate

# Connect to Weaviate (skip gRPC checks)
client = weaviate.connect_to_local(skip_init_checks=True)

# Get the collection
collection = client.collections.get("AbstractSentence")

# Example: semantic search for "REBOA"
results = collection.query.near_text(
    query="REBOA",
    limit=5,
    return_metadata=["score"]
)

print("Top 5 results for 'REBOA':")
for obj in results.objects:
    print(f"Score: {obj.metadata.score:.3f}")
    print(f"Sentence: {obj.properties['sentence']}")
    print(f"Filename: {obj.properties['filename']}")
    print(f"Page: {obj.properties['page']}")
    print()

# Example: semantic search for "sudbury vertigo score"
results = collection.query.near_text(
    query="sudbury vertigo score",
    limit=5,
    return_metadata=["score"]
)

print("Top 5 results for 'sudbury vertigo score':")
for obj in results.objects:
    print(f"Score: {obj.metadata.score:.3f}")
    print(f"Sentence: {obj.properties['sentence']}")
    print(f"Filename: {obj.properties['filename']}")
    print(f"Page: {obj.properties['page']}")
    print()

client.close()
