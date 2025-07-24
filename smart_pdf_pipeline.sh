#!/bin/bash

# Smart PDF Processing Script with Duplicate Prevention
# This script intelligently processes only new or changed PDFs

set -e  # Exit on any error

echo "🏥 Smart Medical PDF Processing Pipeline"
echo "========================================"

# Configuration
ABSTRACTS_DIR="$HOME/abstracts"
JSONL_FILE="processed_abstracts_sentences.jsonl"
TRACKING_FILE="processed_files_tracking.json"

# Check if abstracts directory exists
if [ ! -d "$ABSTRACTS_DIR" ]; then
    echo "❌ Abstracts directory not found: $ABSTRACTS_DIR"
    echo "💡 Create it with: mkdir -p $ABSTRACTS_DIR"
    exit 1
fi

# Count PDF files
PDF_COUNT=$(find "$ABSTRACTS_DIR" -name "*.pdf" | wc -l)
echo "📚 Found $PDF_COUNT PDF files in $ABSTRACTS_DIR"

if [ "$PDF_COUNT" -eq 0 ]; then
    echo "ℹ️  No PDF files to process"
    exit 0
fi

# Check for force rebuild flag
FORCE_REBUILD=false
if [ "$1" = "--force" ] || [ "$1" = "--full-rebuild" ]; then
    FORCE_REBUILD=true
    echo "🔄 Force rebuild mode enabled"
fi

# Step 1: Smart PDF Processing
echo ""
echo "🔍 Step 1: Analyzing PDFs for processing..."

if [ "$FORCE_REBUILD" = true ]; then
    echo "🔄 Force rebuild: Processing ALL PDFs..."
    python process_abstracts.py
    PROCESSING_EXIT_CODE=$?
else
    echo "🧠 Smart mode: Processing only new/changed PDFs..."
    python process_abstracts_incremental.py
    PROCESSING_EXIT_CODE=$?
fi

if [ $PROCESSING_EXIT_CODE -ne 0 ]; then
    echo "❌ PDF processing failed with exit code $PROCESSING_EXIT_CODE"
    exit 1
fi

# Check if JSONL file was created/updated
if [ ! -f "$JSONL_FILE" ]; then
    echo "❌ No output file generated. Nothing to upload."
    exit 1
fi

# Count sentences in JSONL
SENTENCE_COUNT=$(wc -l < "$JSONL_FILE")
echo "📊 Generated $SENTENCE_COUNT sentences in $JSONL_FILE"

# Step 2: Smart Weaviate Upload
echo ""
echo "📤 Step 2: Uploading to Weaviate..."

# Check if Weaviate is running
if ! curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; then
    echo "❌ Weaviate is not running on localhost:8080"
    echo "💡 Start Weaviate with: docker run -d --name weaviate -p 8080:8080 cr.weaviate.io/semitechnologies/weaviate:latest"
    exit 1
fi

if [ "$FORCE_REBUILD" = true ]; then
    echo "🔄 Force rebuild: Recreating entire Weaviate collection..."
    python upload_to_weaviate_incremental.py --full-rebuild
else
    echo "➕ Smart mode: Incremental update to Weaviate..."
    python upload_to_weaviate_incremental.py
fi

UPLOAD_EXIT_CODE=$?
if [ $UPLOAD_EXIT_CODE -ne 0 ]; then
    echo "❌ Weaviate upload failed with exit code $UPLOAD_EXIT_CODE"
    exit 1
fi

# Step 3: Generate Report
echo ""
echo "📋 Processing Report"
echo "==================="

if [ -f "$TRACKING_FILE" ]; then
    TRACKED_FILES=$(python3 -c "
import json
with open('$TRACKING_FILE', 'r') as f:
    data = json.load(f)
    print(len(data))
    for filename, info in data.items():
        print(f'  • {filename} ({info.get(\"sentence_count\", \"?\")}) sentences)')
")
    echo "📁 Tracked files:"
    echo "$TRACKED_FILES"
else
    echo "⚠️  No tracking file found"
fi

echo ""
echo "✅ Pipeline completed successfully!"
echo ""
echo "🎯 Next Steps:"
echo "  • Test new content: python medical_rag_chatbot.py"
echo "  • Add more PDFs: Copy to $ABSTRACTS_DIR and run this script again"
echo "  • Force full rebuild: $0 --force"
echo ""
echo "💡 Tip: This script only processes new/changed PDFs for efficiency!"
