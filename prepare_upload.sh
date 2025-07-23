#!/bin/bash

# Repository Upload Preparation Script
# This script prepares the emarag-weaviate repository for GitHub upload

echo "🚀 Preparing emarag-weaviate repository for GitHub upload..."
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "medical_rag_chatbot.py" ]; then
    echo "❌ Please run this script from the emarag-weaviate directory"
    exit 1
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "📁 Adding files to Git..."
git add .

# Check git status
echo "📊 Git status:"
git status --short

# Create initial commit
echo ""
read -p "Enter commit message (default: 'Initial commit - Emergency Medicine RAG Chatbot'): " commit_message
if [ -z "$commit_message" ]; then
    commit_message="Initial commit - Emergency Medicine RAG Chatbot"
fi

git commit -m "$commit_message"

echo "✅ Initial commit created"

# Check for .env file and warn
if [ -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: .env file detected!"
    echo "Make sure it's excluded by .gitignore and doesn't contain sensitive data"
    echo "Check the file contents:"
    echo "----------------------------------------"
    cat .env
    echo "----------------------------------------"
    echo ""
    read -p "Continue? (y/n): " continue_upload
    if [ "$continue_upload" != "y" ]; then
        echo "❌ Upload cancelled"
        exit 1
    fi
fi

# Repository information
echo ""
echo "📋 Repository Information:"
echo "Name: emarag-weaviate"
echo "Description: Emergency Medicine RAG Chatbot with Weaviate vector database"
echo "Main files:"
echo "  - medical_rag_chatbot.py (Main application)"
echo "  - README.md (Documentation)"
echo "  - setup.sh (Installation script)"
echo "  - docker-compose.yml (Weaviate database)"
echo ""

# Generate tree structure
echo "📁 Project Structure:"
echo "emarag-weaviate/"
echo "├── medical_rag_chatbot.py      # Main chatbot application"
echo "├── upload_to_weaviate.py       # Data ingestion script"
echo "├── process_abstracts.py        # Abstract processing utilities"
echo "├── sample_query.py             # Example usage script"
echo "├── setup.sh                    # Automated setup script"
echo "├── start_chatbot.sh           # Launch script"
echo "├── docker-compose.yml          # Weaviate database setup"
echo "├── requirements.txt            # Python dependencies"
echo "├── .env.example               # Environment template"
echo "├── README.md                  # Project documentation"
echo "├── LICENSE                    # MIT license"
echo "├── CHANGELOG.md               # Version history"
echo "├── CONTRIBUTING.md            # Contribution guidelines"
echo "├── .gitignore                 # Git ignore rules"
echo "├── processed_abstracts.jsonl  # Medical abstracts data"
echo "├── processed_abstracts_sentences.jsonl  # Sentence-level data"
echo "├── .github/workflows/         # GitHub Actions (CI/CD)"
echo "├── examples/                  # Usage examples"
echo "├── tests/                     # Test scripts"
echo "└── venv312/                   # Virtual environment (excluded)"

echo ""
echo "🎯 Next Steps to Upload to GitHub:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: emarag-weaviate"
echo "   - Description: Emergency Medicine RAG Chatbot with Weaviate vector database"
echo "   - Public or Private (your choice)"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/YOURUSERNAME/emarag-weaviate.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📝 Repository Features:"
echo "✅ Complete medical RAG chatbot with dual LLM support"
echo "✅ Docker-based Weaviate vector database"
echo "✅ Comprehensive documentation and setup scripts"
echo "✅ GitHub Actions CI/CD pipeline"
echo "✅ Example queries and test scripts"
echo "✅ Medical literature data included"
echo "✅ MIT license with medical disclaimer"
echo ""
echo "🏥 Ready for medical AI research and education!"

# Final git status
echo ""
echo "📊 Final repository status:"
git log --oneline -5 || echo "No commits yet"
echo ""
echo "📁 Files ready for upload:"
git ls-files | wc -l | xargs echo "Total files:"

echo ""
echo "🎉 Repository is ready for GitHub upload!"
