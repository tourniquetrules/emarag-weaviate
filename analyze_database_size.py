#!/usr/bin/env python3
"""
Comprehensive Weaviate Database Size Analysis
Analyzes the size and storage requirements of your Weaviate vector database.
"""

import weaviate
import json
import sys
import subprocess
import os
from datetime import datetime

def get_docker_container_stats():
    """Get Docker container resource usage statistics"""
    try:
        # Get container stats
        result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 
                               'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("🐳 Docker Container Stats:")
            print("=" * 50)
            print(result.stdout)
        else:
            print("⚠️ Could not retrieve Docker stats")
    except Exception as e:
        print(f"⚠️ Error getting Docker stats: {e}")

def get_weaviate_volume_size():
    """Get the size of the Weaviate Docker volume"""
    try:
        # Get volume info
        result = subprocess.run(['docker', 'exec', 'emarag-weaviate-weaviate-1', 
                               'du', '-sh', '/var/lib/weaviate'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            size = result.stdout.strip().split('\t')[0]
            print(f"💾 Weaviate Data Directory Size: {size}")
            return size
        else:
            print("⚠️ Could not get Weaviate volume size")
            return "Unknown"
    except Exception as e:
        print(f"⚠️ Error getting volume size: {e}")
        return "Unknown"

def get_detailed_weaviate_stats():
    """Get detailed statistics from Weaviate"""
    try:
        client = weaviate.connect_to_local(skip_init_checks=True)
        print("✅ Connected to Weaviate")
        
        # Get collection info
        collection = client.collections.get("AbstractSentence")
        
        # Get total object count
        total_objects = collection.aggregate.over_all(total_count=True).total_count
        
        # Sample some objects to estimate average sizes
        sample_results = collection.query.fetch_objects(limit=10)
        
        # Calculate average text length
        total_text_length = 0
        object_count = 0
        
        for obj in sample_results.objects:
            if 'sentence' in obj.properties:
                total_text_length += len(obj.properties['sentence'])
                object_count += 1
        
        avg_text_length = total_text_length / object_count if object_count > 0 else 0
        
        # Estimate storage requirements
        estimated_text_storage = (total_objects * avg_text_length) / (1024 * 1024)  # MB
        
        # Vector dimensions (assuming standard embedding size)
        vector_dimensions = 768  # Common for many models
        bytes_per_float = 4
        estimated_vector_storage = (total_objects * vector_dimensions * bytes_per_float) / (1024 * 1024)  # MB
        
        print("\n📊 Weaviate Database Analysis:")
        print("=" * 50)
        print(f"📄 Total Objects (Sentences): {total_objects:,}")
        print(f"📝 Average Text Length: {avg_text_length:.1f} characters")
        print(f"📐 Estimated Vector Dimensions: {vector_dimensions}")
        print(f"💭 Estimated Text Storage: {estimated_text_storage:.2f} MB")
        print(f"🔢 Estimated Vector Storage: {estimated_vector_storage:.2f} MB")
        print(f"📦 Total Estimated Storage: {estimated_text_storage + estimated_vector_storage:.2f} MB")
        
        # Try to get schema info (this might fail with newer Weaviate client)
        try:
            # Use the new client API for schema
            collections = client.collections.list_all()
            
            print(f"\n🏗️ Schema Information:")
            print("=" * 30)
            for collection_name in collections:
                try:
                    collection = client.collections.get(collection_name)
                    print(f"📋 Collection: {collection_name}")
                    print(f"   � Estimated objects: Part of {total_objects:,} total")
                except Exception as schema_e:
                    print(f"   ⚠️ Could not get details: {schema_e}")
        except Exception as schema_error:
            print(f"\n⚠️ Schema information unavailable with current client: {schema_error}")
        
        client.close()
        return {
            'total_objects': total_objects,
            'avg_text_length': avg_text_length,
            'estimated_text_storage_mb': estimated_text_storage,
            'estimated_vector_storage_mb': estimated_vector_storage,
            'total_estimated_storage_mb': estimated_text_storage + estimated_vector_storage
        }
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        return None

def get_pdf_source_size():
    """Get the total size of source PDF files"""
    try:
        abstracts_path = "/home/tourniquetrules/abstracts"
        if os.path.exists(abstracts_path):
            result = subprocess.run(['du', '-sh', abstracts_path], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                size = result.stdout.strip().split('\t')[0]
                print(f"📁 Source PDFs Directory Size: {size}")
                return size
            else:
                print("⚠️ Could not get PDFs directory size")
                return "Unknown"
        else:
            print("⚠️ Abstracts directory not found")
            return "Not found"
    except Exception as e:
        print(f"⚠️ Error getting PDFs size: {e}")
        return "Unknown"

def main():
    print("🔍 Weaviate Database Size Analysis")
    print("=" * 60)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get Docker stats
    get_docker_container_stats()
    print()
    
    # Get volume size
    volume_size = get_weaviate_volume_size()
    print()
    
    # Get source PDF size
    pdf_size = get_pdf_source_size()
    print()
    
    # Get detailed Weaviate statistics
    weaviate_stats = get_detailed_weaviate_stats()
    
    # Summary
    print("\n📋 Size Summary:")
    print("=" * 30)
    print(f"💾 Weaviate Container Data: {volume_size}")
    print(f"📁 Source PDF Files: {pdf_size}")
    
    if weaviate_stats:
        print(f"🔢 Database Objects: {weaviate_stats['total_objects']:,} sentences")
        print(f"📦 Estimated Storage: {weaviate_stats['total_estimated_storage_mb']:.1f} MB")
        
        # Storage efficiency
        try:
            volume_mb = float(volume_size.replace('M', '').replace('G', '000')) if 'M' in volume_size or 'G' in volume_size else 0
            if volume_mb > 0:
                efficiency = (weaviate_stats['total_estimated_storage_mb'] / volume_mb) * 100
                print(f"📊 Storage Efficiency: {efficiency:.1f}%")
        except:
            print("📊 Storage Efficiency: Could not calculate")
    
    print("\n💡 Tips:")
    print("  • Vector storage typically dominates text storage")
    print("  • Weaviate includes indexing overhead and metadata")
    print("  • Compression and optimization reduce actual disk usage")
    print("  • Consider vector dimensions vs. accuracy trade-offs")

if __name__ == "__main__":
    main()
