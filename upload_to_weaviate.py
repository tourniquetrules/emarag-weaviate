import weaviate
import json

# Path to your sentence-level JSONL file

import weaviate
import json
from weaviate.collections.classes.config import Property, DataType, Configure
import os

# Path to your sentence-level JSONL file
jsonl_path = "processed_abstracts_sentences.jsonl"

# Connect to your Weaviate instance (adjust URL and API key as needed)
client = weaviate.connect_to_local(skip_init_checks=True)

# Define the collection schema if not already present in Weaviate
if client.collections.exists("AbstractSentence"):
    print("Deleting existing AbstractSentence collection...")
    client.collections.delete("AbstractSentence")
    print("Collection deleted.")
print("Creating AbstractSentence collection with text2vec-transformers vectorizer...")

client.collections.create(
    name="AbstractSentence",
    description="A sentence chunk from a medical/scientific abstract.",
    properties=[
        Property(name="filename", data_type=DataType.TEXT),
        Property(name="page", data_type=DataType.INT),
        Property(name="block_label", data_type=DataType.TEXT),
        Property(name="sentence", data_type=DataType.TEXT),
        Property(name="entities", data_type=DataType.TEXT_ARRAY),
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
)
print("Collection created.")

# Upload each sentence as an object
def upload_sentences(jsonl_path):
    collection = client.collections.get("AbstractSentence")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # Convert entities to just the entity text for Weaviate (optional)
            record["entities"] = [ent[0] for ent in record.get("entities", [])]
            # Convert page to int if possible
            if record.get("page") is not None:
                try:
                    record["page"] = int(record["page"])
                except Exception:
                    record["page"] = None
            collection.data.insert(record)
    print("Upload complete.")

if __name__ == "__main__":
    upload_sentences(jsonl_path)
