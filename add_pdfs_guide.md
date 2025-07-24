# 📚 Adding PDFs to Weaviate Database - Complete Guide

## Overview
Your Weaviate medical RAG chatbot uses a 3-step process to add new PDFs:
1. **Process PDFs** → Extract and chunk text with spaCy
2. **Upload to Weaviate** → Store processed chunks in vector database
3. **Query Ready** → New content available for RAG queries

## Current Pipeline Components

### 1. PDF Processing Script: `process_abstracts.py`
- **Location**: `/home/tourniquetrules/emarag-weaviate/process_abstracts.py`
- **Input**: PDF files in `~/abstracts/` directory
- **Output**: `processed_abstracts_sentences.jsonl` file
- **Features**:
  - Uses spaCy SciSciBERT model for medical text processing
  - Extracts sentences with medical entity recognition
  - Preserves filename, page number, and layout information
  - GPU acceleration support

### 2. Weaviate Upload Script: `upload_to_weaviate.py`
- **Location**: `/home/tourniquetrules/emarag-weaviate/upload_to_weaviate.py`
- **Input**: `processed_abstracts_sentences.jsonl`
- **Output**: Data loaded into Weaviate's "AbstractSentence" collection
- **Features**:
  - Creates/recreates collection schema
  - Uses text2vec-transformers vectorizer
  - Preserves metadata (filename, page, entities)

## 🚀 Quick Start: Adding New PDFs

### Step 1: Place PDFs in the abstracts directory
```bash
# Make sure abstracts directory exists
mkdir -p ~/abstracts

# Copy your new PDF files to the abstracts directory
cp /path/to/your/new_medical_paper.pdf ~/abstracts/
cp /path/to/another_paper.pdf ~/abstracts/

# Verify files are there
ls -la ~/abstracts/*.pdf
```

### Step 2: Process the PDFs
```bash
cd /home/tourniquetrules/emarag-weaviate

# Run the PDF processing script
python process_abstracts.py

# This will create/update: processed_abstracts_sentences.jsonl
```

### Step 3: Upload to Weaviate
```bash
# Make sure Weaviate is running
docker ps | grep weaviate

# Upload processed sentences to Weaviate
python upload_to_weaviate.py
```

### Step 4: Verify Upload
```bash
# Start your chatbot to test new content
python medical_rag_chatbot.py

# Test with queries related to your new PDFs
```

## 🔧 Advanced: Modifying the Processing Pipeline

### Adding More PDF Sources
You can modify `process_abstracts.py` to process PDFs from multiple directories:

```python
# Edit process_abstracts.py
abstracts_dirs = [
    os.path.expanduser("~/abstracts"),
    os.path.expanduser("~/medical_papers"),
    os.path.expanduser("~/emergency_medicine_docs")
]

for abstracts_dir in abstracts_dirs:
    if os.path.exists(abstracts_dir):
        for filename in os.listdir(abstracts_dir):
            if filename.endswith(".pdf"):
                # ... rest of processing code
```

### Incremental Updates (Add Without Deleting)
To add new PDFs without deleting existing data, modify `upload_to_weaviate.py`:

```python
# Comment out the deletion part
# if client.collections.exists("AbstractSentence"):
#     print("Deleting existing AbstractSentence collection...")
#     client.collections.delete("AbstractSentence")

# Add logic to only create collection if it doesn't exist
if not client.collections.exists("AbstractSentence"):
    print("Creating AbstractSentence collection...")
    client.collections.create(...)
else:
    print("Using existing AbstractSentence collection...")
```

## 📋 Troubleshooting

### Common Issues

1. **Weaviate Connection Failed**
   ```bash
   # Check if Weaviate is running
   docker ps | grep weaviate
   
   # Start Weaviate if not running
   docker run -d --name weaviate -p 8080:8080 \
     cr.weaviate.io/semitechnologies/weaviate:latest
   ```

2. **spaCy Model Missing**
   ```bash
   # Install the scientific spaCy model
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
   ```

3. **Memory Issues with Large PDFs**
   ```python
   # Process PDFs in smaller batches
   # Modify process_abstracts.py to process files one at a time
   ```

## 📊 Monitoring Your Database

### Check Collection Status
```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("AbstractSentence")

# Get total count
total_objects = collection.aggregate.over_all(total_count=True)
print(f"Total documents in database: {total_objects.total_count}")

# Get sample objects
objects = collection.query.fetch_objects(limit=5)
for obj in objects.objects:
    print(f"File: {obj.properties['filename']}")
    print(f"Text: {obj.properties['sentence'][:100]}...")
    print("---")
```

## 🎯 Best Practices

1. **Organize Your PDFs**
   - Use descriptive filenames
   - Group related papers in subdirectories
   - Keep originals in a separate backup location

2. **Monitor Processing**
   - Check the output logs for any processing errors
   - Verify sentence count in the JSONL file
   - Test queries after each upload

3. **Database Maintenance**
   - Periodically check collection size
   - Monitor query performance
   - Consider rebuilding collection if performance degrades

4. **Quality Control**
   - Review extracted text for accuracy
   - Test edge cases with new document types
   - Validate entity extraction results

## 🔄 Automation Script

Here's a complete script to automate the entire process:

```bash
#!/bin/bash
# save as: add_new_pdfs.sh

echo "🏥 Adding new PDFs to Medical RAG Database"
echo "=========================================="

# Step 1: Check if abstracts directory has new files
NEW_PDFS=$(find ~/abstracts -name "*.pdf" -newer processed_abstracts_sentences.jsonl 2>/dev/null | wc -l)

if [ "$NEW_PDFS" -eq 0 ]; then
    echo "ℹ️  No new PDFs found since last processing"
    exit 0
fi

echo "📚 Found $NEW_PDFS new PDF(s) to process"

# Step 2: Process PDFs
echo "🔄 Processing PDFs..."
python process_abstracts.py

if [ $? -ne 0 ]; then
    echo "❌ PDF processing failed"
    exit 1
fi

# Step 3: Upload to Weaviate
echo "📤 Uploading to Weaviate..."
python upload_to_weaviate.py

if [ $? -ne 0 ]; then
    echo "❌ Weaviate upload failed"
    exit 1
fi

echo "✅ Successfully added new PDFs to the database!"
echo "🚀 You can now restart your medical_rag_chatbot.py to use the new content"
```

## Summary

To add new PDFs to your Weaviate database:

1. **Copy PDFs** to `~/abstracts/` directory
2. **Run** `python process_abstracts.py` 
3. **Run** `python upload_to_weaviate.py`
4. **Test** the new content in your chatbot

The current pipeline is well-designed for medical documents and preserves important metadata for accurate retrieval and source attribution.
