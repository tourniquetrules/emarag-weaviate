#!/usr/bin/env python3
"""
Check what's currently in the Weaviate database and create tracking file
"""

import weaviate
import json
import os
from collections import defaultdict

def check_weaviate_status():
    """Check what files are currently in Weaviate and create tracking data"""
    
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(skip_init_checks=True)
        print("✅ Connected to Weaviate")
        
        # Check if collection exists
        if not client.collections.exists("AbstractSentence"):
            print("❌ AbstractSentence collection does not exist")
            return False
        
        collection = client.collections.get("AbstractSentence")
        print("✅ Found AbstractSentence collection")
        
        # Get total count
        try:
            total_result = collection.aggregate.over_all(total_count=True)
            total_count = total_result.total_count
            print(f"📊 Total sentences in database: {total_count}")
        except Exception as e:
            print(f"⚠️  Could not get total count: {e}")
            total_count = "Unknown"
        
        # Get all objects to analyze files
        print("🔍 Analyzing files in database...")
        
        file_stats = defaultdict(lambda: {"count": 0, "pages": set()})
        
        # Query in batches to get all data
        limit = 1000
        offset = 0
        all_processed = False
        
        while not all_processed:
            try:
                results = collection.query.fetch_objects(
                    limit=limit,
                    offset=offset
                )
                
                if not results.objects:
                    all_processed = True
                    break
                
                for obj in results.objects:
                    filename = obj.properties.get('filename', 'Unknown')
                    page = obj.properties.get('page', 'N/A')
                    
                    file_stats[filename]["count"] += 1
                    if page != 'N/A' and page is not None:
                        file_stats[filename]["pages"].add(page)
                
                offset += limit
                print(f"  📝 Processed {offset} sentences...")
                
                # Safety check to avoid infinite loop
                if offset > 50000:  # Reasonable limit
                    print("⚠️  Reached safety limit, stopping scan")
                    break
                    
            except Exception as e:
                print(f"❌ Error fetching batch at offset {offset}: {e}")
                break
        
        # Display results
        print(f"\n📋 Files in Weaviate Database:")
        print("=" * 60)
        
        for filename, stats in sorted(file_stats.items()):
            pages = sorted(list(stats["pages"])) if stats["pages"] else ["N/A"]
            page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 and pages[0] != "N/A" else str(pages[0]) if pages else "N/A"
            print(f"📄 {filename}")
            print(f"   Sentences: {stats['count']}, Pages: {page_range}")
        
        print(f"\n📊 Summary:")
        print(f"  • Total files in database: {len(file_stats)}")
        print(f"  • Total sentences: {total_count}")
        
        # Check against abstracts directory
        abstracts_dir = os.path.expanduser("~/abstracts")
        if os.path.exists(abstracts_dir):
            pdf_files = [f for f in os.listdir(abstracts_dir) if f.endswith('.pdf')]
            print(f"  • PDF files in {abstracts_dir}: {len(pdf_files)}")
            
            # Find which files are missing from database
            db_files = set(file_stats.keys())
            disk_files = set(pdf_files)
            
            missing_from_db = disk_files - db_files
            extra_in_db = db_files - disk_files
            
            if missing_from_db:
                print(f"\n⚠️  Files on disk but NOT in database ({len(missing_from_db)}):")
                for f in sorted(missing_from_db):
                    print(f"  • {f}")
            
            if extra_in_db:
                print(f"\n⚠️  Files in database but NOT on disk ({len(extra_in_db)}):")
                for f in sorted(extra_in_db):
                    print(f"  • {f}")
            
            if not missing_from_db and not extra_in_db:
                print(f"\n✅ All {len(pdf_files)} PDF files are properly loaded in database!")
                
                # Create tracking file for future use
                print(f"\n🔄 Creating tracking file for future incremental updates...")
                create_tracking_file(abstracts_dir, pdf_files, file_stats)
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        return False

def create_tracking_file(abstracts_dir, pdf_files, file_stats):
    """Create a tracking file based on current state"""
    import hashlib
    from datetime import datetime
    
    tracking_data = {}
    
    for filename in pdf_files:
        pdf_path = os.path.join(abstracts_dir, filename)
        if os.path.exists(pdf_path):
            # Get file hash and stats
            file_stat = os.stat(pdf_path)
            
            # Calculate MD5 hash
            hash_md5 = hashlib.md5()
            with open(pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            tracking_data[filename] = {
                "hash": hash_md5.hexdigest(),
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime,
                "processed_date": datetime.now().isoformat(),
                "sentence_count": file_stats.get(filename, {}).get("count", 0),
                "status": "loaded_in_database"
            }
    
    # Save tracking file
    tracking_path = "processed_files_tracking.json"
    with open(tracking_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"✅ Created tracking file: {tracking_path}")
    print(f"   📊 Tracking {len(tracking_data)} files")

if __name__ == "__main__":
    print("🔍 Checking Weaviate Database Status...")
    check_weaviate_status()
