import weaviate
import json
import os
from weaviate.collections.classes.config import Property, DataType, Configure
from datetime import datetime

# Path to your sentence-level JSONL file
jsonl_path = "processed_abstracts_sentences.jsonl"
tracking_path = "processed_files_tracking.json"

def load_tracking_data():
    """Load tracking data of processed files"""
    if os.path.exists(tracking_path):
        with open(tracking_path, "r") as f:
            return json.load(f)
    return {}

def get_existing_filenames(collection):
    """Get list of filenames already in Weaviate collection"""
    try:
        # Query for all unique filenames
        result = collection.aggregate.over_all(
            group_by="filename"
        )
        existing_files = set()
        if hasattr(result, 'groups') and result.groups:
            for group in result.groups:
                if group.grouped_by and 'value' in group.grouped_by:
                    existing_files.add(group.grouped_by['value'])
        return existing_files
    except Exception as e:
        print(f"Could not retrieve existing filenames: {e}")
        return set()

def upload_sentences_incremental(jsonl_path, force_full_rebuild=False):
    """Upload sentences with incremental update support"""
    
    # Connect to Weaviate
    client = weaviate.connect_to_local(skip_init_checks=True)
    
    # Check if collection exists
    collection_exists = client.collections.exists("AbstractSentence")
    
    if force_full_rebuild or not collection_exists:
        # Full rebuild mode
        if collection_exists:
            print("🗑️  Deleting existing AbstractSentence collection...")
            client.collections.delete("AbstractSentence")
            print("Collection deleted.")
        
        print("🔨 Creating AbstractSentence collection with text2vec-transformers vectorizer...")
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
        
        # Upload all data
        collection = client.collections.get("AbstractSentence")
        total_count = 0
        
        print("📤 Uploading all sentences...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                record["entities"] = [ent[0] for ent in record.get("entities", [])]
                if record.get("page") is not None:
                    try:
                        record["page"] = int(record["page"])
                    except Exception:
                        record["page"] = None
                collection.data.insert(record)
                total_count += 1
        
        print(f"✅ Full upload complete. Added {total_count} sentences.")
        
    else:
        # Incremental update mode
        print("🔄 Incremental update mode")
        collection = client.collections.get("AbstractSentence")
        
        # Get existing filenames in Weaviate
        existing_files = get_existing_filenames(collection)
        print(f"📊 Found {len(existing_files)} files already in database")
        
        # Load tracking data to see what files were just processed
        tracking_data = load_tracking_data()
        
        # Read JSONL and only upload new files
        new_files_found = set()
        sentences_to_upload = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                filename = record["filename"]
                
                # Check if this file is new or updated
                if filename not in existing_files:
                    new_files_found.add(filename)
                    record["entities"] = [ent[0] for ent in record.get("entities", [])]
                    if record.get("page") is not None:
                        try:
                            record["page"] = int(record["page"])
                        except Exception:
                            record["page"] = None
                    sentences_to_upload.append(record)
        
        if sentences_to_upload:
            print(f"📤 Uploading {len(sentences_to_upload)} sentences from {len(new_files_found)} new files:")
            for filename in new_files_found:
                print(f"  • {filename}")
            
            # Upload new sentences
            for record in sentences_to_upload:
                collection.data.insert(record)
            
            print(f"✅ Incremental upload complete. Added {len(sentences_to_upload)} new sentences.")
        else:
            print("ℹ️  No new files to upload. Database is up to date.")
    
    # Print final statistics
    try:
        total_objects = collection.aggregate.over_all(total_count=True)
        print(f"📊 Total sentences in database: {total_objects.total_count}")
    except:
        print("📊 Could not retrieve total count")
    
    client.close()

def main():
    """Main upload function with options"""
    import sys
    
    force_rebuild = "--full-rebuild" in sys.argv or "--force" in sys.argv
    
    if force_rebuild:
        print("🔄 Force full rebuild requested")
        upload_sentences_incremental(jsonl_path, force_full_rebuild=True)
    else:
        print("➕ Incremental update mode (use --full-rebuild to force complete rebuild)")
        upload_sentences_incremental(jsonl_path, force_full_rebuild=False)

if __name__ == "__main__":
    main()
