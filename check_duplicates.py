#!/usr/bin/env python3
"""
Check for duplicate sentences in Weaviate database
"""
import weaviate
from collections import defaultdict

def check_duplicates():
    """Check for duplicate sentences in the database"""
    print("🔍 Checking for duplicates in Weaviate database...")
    
    client = weaviate.connect_to_local(skip_init_checks=True)
    collection = client.collections.get("AbstractSentence")
    
    sentence_counts = defaultdict(int)
    filename_sentence_pairs = defaultdict(int)
    total_processed = 0
    
    # Use pagination to get all objects
    batch_size = 1000
    offset = 0
    
    while True:
        try:
            # Get batch of objects
            results = collection.query.fetch_objects(
                limit=batch_size,
                offset=offset,
                return_properties=["filename", "sentence"]
            )
            
            if not results.objects:
                break
                
            print(f"📝 Processing batch {offset}-{offset + len(results.objects)}")
            
            for obj in results.objects:
                if obj.properties:
                    sentence = obj.properties.get("sentence", "")
                    filename = obj.properties.get("filename", "")
                    
                    sentence_counts[sentence] += 1
                    filename_sentence_pairs[(filename, sentence)] += 1
                    total_processed += 1
            
            offset += batch_size
            
            # Break if we got fewer results than requested (end of data)
            if len(results.objects) < batch_size:
                break
                
        except Exception as e:
            print(f"⚠️  Error at offset {offset}: {e}")
            break
    
    print(f"📊 Analysis Results:")
    print(f"  • Total sentences processed: {total_processed}")
    print(f"  • Unique sentences: {len(sentence_counts)}")
    
    # Find duplicates
    duplicates = {sent: count for sent, count in sentence_counts.items() if count > 1}
    file_duplicates = {pair: count for pair, count in filename_sentence_pairs.items() if count > 1}
    
    print(f"  • Duplicate sentences: {len(duplicates)}")
    print(f"  • File-sentence pair duplicates: {len(file_duplicates)}")
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate sentences:")
        duplicate_total = sum(duplicates.values()) - len(duplicates)  # Extra copies
        print(f"  • Total duplicate entries: {duplicate_total}")
        
        print("  • Top duplicates:")
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        for i, (sent, count) in enumerate(sorted_duplicates[:5]):
            print(f"    {i+1}. Count: {count} - {sent[:80]}...")
    else:
        print("✅ No duplicate sentences found!")
    
    if file_duplicates:
        print(f"\n⚠️  Found {len(file_duplicates)} file-sentence duplicates:")
        for i, ((filename, sent), count) in enumerate(list(file_duplicates.items())[:3]):
            print(f"  {i+1}. Count: {count}")
            print(f"     File: {filename}")
            print(f"     Text: {sent[:60]}...")
    else:
        print("✅ No file-sentence duplicates found!")
    
    # Calculate duplicate percentage
    if total_processed > 0:
        duplicate_percentage = (len(duplicates) / len(sentence_counts)) * 100
        print(f"\n📊 Duplicate Statistics:")
        print(f"  • Duplicate rate: {duplicate_percentage:.1f}%")
        if duplicates:
            total_extra_entries = sum(duplicates.values()) - len(duplicates)
            space_waste_percentage = (total_extra_entries / total_processed) * 100
            print(f"  • Storage waste from duplicates: {space_waste_percentage:.1f}%")
    
    client.close()
    
    return {
        "total_processed": total_processed,
        "unique_sentences": len(sentence_counts),
        "duplicates": len(duplicates),
        "file_duplicates": len(file_duplicates),
        "has_duplicates": len(duplicates) > 0 or len(file_duplicates) > 0
    }

if __name__ == "__main__":
    check_duplicates()
