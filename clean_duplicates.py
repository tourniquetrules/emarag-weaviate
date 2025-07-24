#!/usr/bin/env python3
"""
Clean up duplicate entries in Weaviate database
This script will:
1. Find all unique sentences 
2. Delete the entire collection
3. Re-upload only unique data from the JSONL file
"""
import weaviate
import json
import os
from collections import defaultdict
from weaviate.collections.classes.config import Property, DataType, Configure

def clean_duplicates():
    """Remove duplicates from Weaviate database"""
    print("🧹 Cleaning Duplicate Data from Weaviate Database")
    print("=" * 60)
    
    # Load data from JSONL file
    jsonl_path = "processed_abstracts_sentences.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL file not found: {jsonl_path}")
        return False
    
    print("📂 Loading data from JSONL file...")
    sentence_dict = {}  # Use dict to automatically deduplicate
    total_lines = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            record = json.loads(line)
            sentence = record.get("sentence", "")
            
            # Use sentence text as key to deduplicate
            # Keep the first occurrence of each sentence
            if sentence and sentence not in sentence_dict:
                sentence_dict[sentence] = record
    
    unique_records = list(sentence_dict.values())
    
    print(f"📊 Deduplication Results:")
    print(f"  • Total lines in JSONL: {total_lines:,}")
    print(f"  • Unique sentences: {len(unique_records):,}")
    print(f"  • Duplicates removed: {total_lines - len(unique_records):,}")
    print(f"  • Space saved: {((total_lines - len(unique_records)) / total_lines * 100):.1f}%")
    
    # Connect to Weaviate
    print("\n🔗 Connecting to Weaviate...")
    client = weaviate.connect_to_local(skip_init_checks=True)
    
    # Delete existing collection
    if client.collections.exists("AbstractSentence"):
        print("🗑️  Deleting existing AbstractSentence collection...")
        client.collections.delete("AbstractSentence")
        print("Collection deleted.")
    
    # Recreate collection
    print("🔨 Creating clean AbstractSentence collection...")
    client.collections.create(
        name="AbstractSentence",
        description="A sentence chunk from a medical/scientific abstract (deduplicated).",
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
    
    # Upload clean data
    collection = client.collections.get("AbstractSentence")
    print(f"📤 Uploading {len(unique_records):,} unique sentences...")
    
    uploaded_count = 0
    for record in unique_records:
        try:
            # Clean up record format
            clean_record = {
                "filename": record.get("filename", ""),
                "page": record.get("page"),
                "block_label": record.get("block_label", ""),
                "sentence": record.get("sentence", ""),
                "entities": [ent[0] if isinstance(ent, list) else str(ent) 
                            for ent in record.get("entities", [])]
            }
            
            # Ensure page is int or None
            if clean_record["page"] is not None:
                try:
                    clean_record["page"] = int(clean_record["page"])
                except:
                    clean_record["page"] = None
            
            collection.data.insert(clean_record)
            uploaded_count += 1
            
            if uploaded_count % 1000 == 0:
                print(f"  📝 Uploaded {uploaded_count:,} sentences...")
                
        except Exception as e:
            print(f"⚠️  Error uploading record: {e}")
    
    # Final verification
    try:
        total_objects = collection.aggregate.over_all(total_count=True)
        final_count = total_objects.total_count
        print(f"\n✅ Cleanup Complete!")
        print(f"  • Final database size: {final_count:,} sentences")
        print(f"  • Reduction: {total_lines - final_count:,} duplicates removed")
        print(f"  • Space efficiency: {(final_count / total_lines * 100):.1f}% of original")
        
    except Exception as e:
        print(f"⚠️  Could not verify final count: {e}")
        print(f"✅ Upload complete. {uploaded_count:,} unique sentences uploaded.")
    
    client.close()
    return True

def backup_jsonl():
    """Create a backup of the original JSONL before cleanup"""
    jsonl_path = "processed_abstracts_sentences.jsonl"
    backup_path = f"{jsonl_path}.backup"
    
    if os.path.exists(jsonl_path) and not os.path.exists(backup_path):
        print(f"💾 Creating backup: {backup_path}")
        os.system(f"cp {jsonl_path} {backup_path}")
        return True
    return False

if __name__ == "__main__":
    # Create backup first
    backup_jsonl()
    
    # Clean duplicates
    success = clean_duplicates()
    
    if success:
        print("\n🎉 Database cleanup completed successfully!")
        print("\n📋 Next Steps:")
        print("  1. Run: python check_database_status.py")
        print("  2. Test the chatbot with clean data")
        print("  3. Future uploads will be properly incremental")
    else:
        print("\n❌ Cleanup failed. Check the error messages above.")
