# Emergency Medicine RAG Chatbot with Weaviate

A sophisticated Retrieval-Augmented Generation (RAG) chatbot specialized in emergency medicine, powered by Weaviate vector database and supporting multiple LLM providers with advanced cross-encoder reranking.

![Medical AI](https://img.shields.io/badge/Medical%20AI-Emergency%20Medicine-red)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Weaviate](https://img.shields.io/badge/Weaviate-Vector%20DB-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Quick Demo

```bash
# Clone and setup
git clone https://github.com/yourusername/emarag-weaviate.git
cd emarag-weaviate
./setup.sh

# Start the chatbot
python medical_rag_chatbot.py

# Visit http://localhost:7871
```

## 🌟 Features

### 🤖 Dual LLM Support
- **LM Studio Integration**: Local models (DeepSeek R1 8B, Gemma 3 12B)
- **OpenAI Integration**: Cloud models (GPT-4.1 Nano, GPT-4o Latest)
- **Model Selection**: Choose specific models through dropdown interface
- **Performance Tracking**: Real-time tokens/second monitoring

### 🧠 Advanced Query Modes
- **📚 RAG Mode (Default)**: Answers based on emergency medicine literature with source citations
- **🚀 General Knowledge Mode**: Use `@llm` prefix for general medical knowledge (bypasses RAG)
- **Cross-Encoder Reranking**: Improved relevance scoring using sentence-transformers

### 🔍 Sophisticated RAG Pipeline
- **Weaviate Vector Database**: Persistent storage with Docker
- **spaCy SciSciBERT**: Medical NLP with GPU acceleration and fallback to CPU
- **Smart Query Enhancement**: Entity extraction and query optimization
- **Cross-Encoder Reranking**: Secondary relevance scoring for better context selection
- **Relevance Scoring**: Context ranking with citation sources and confidence scores

### 📊 Smart PDF Management
- **Incremental Processing**: Only processes new/changed PDFs for efficiency
- **File Tracking**: MD5 hash-based duplicate prevention
- **Automated Pipeline**: `smart_pdf_pipeline.sh` for batch processing
- **Status Reporting**: Comprehensive processing reports and database statistics

### ⚡ Performance & Monitoring
- **Real-time Metrics**: Tokens/second tracking for both providers
- **Configurable Responses**: 1K-4K token slider for response length
- **Performance Comparison**: Side-by-side LLM evaluation
- **Error Handling**: Graceful fallbacks for GPU/model failures

### 🎨 Modern Web Interface
- **Gradio UI**: Clean, responsive web interface on port 7871
- **Live Status Updates**: Model selection and performance display
- **Interactive Examples**: Built-in query suggestions for both modes
- **Citation Sources**: Automatic source attribution with relevance scores

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- NVIDIA GPU (recommended for spaCy acceleration, CPU fallback available)
- OpenAI API key (optional)
- LM Studio running locally (optional)

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/emarag-weaviate.git
cd emarag-weaviate

# Create virtual environment
python -m venv venv312
source venv312/bin/activate  # Linux/Mac
# or
venv312\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_sci_scibert
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings:
# - OPENAI_API_KEY (if using OpenAI)
# - LM_STUDIO_BASE_URL (if using LM Studio)
```

### 4. Start Weaviate Vector Database
Weaviate runs in Docker and provides the vector storage for your RAG system:

```bash
# Start Weaviate with Docker Compose
docker-compose up -d

# Verify Weaviate is running
docker ps | grep weaviate

# Check Weaviate logs (if needed)
docker-compose logs weaviate

# Access Weaviate console (optional)
# Visit http://localhost:8080 in your browser
```

**What Weaviate Does:**
- 🗄️ **Vector Storage**: Stores embeddings of your medical literature sentences
- 🔍 **Semantic Search**: Enables similarity search for RAG retrieval
- 📊 **Persistence**: Data survives container restarts via Docker volumes
- ⚡ **Performance**: Optimized for fast vector queries

### 5. Upload Medical Data
```bash
# Add PDFs to the abstracts directory
cp your_medical_pdfs/* /home/tourniquetrules/abstracts/

# Process and upload using smart pipeline
./smart_pdf_pipeline.sh

# Verify data was uploaded
python check_database_status.py
```

### 6. Launch Chatbot
```bash
python medical_rag_chatbot.py
```

Visit `http://localhost:7871` to access the web interface.

## 📁 Project Structure

```
emarag-weaviate/
├── medical_rag_chatbot.py              # Main chatbot application with cross-encoder reranking
├── smart_pdf_pipeline.sh               # Automated PDF processing pipeline
├── process_abstracts_incremental.py    # Incremental PDF processing with tracking
├── upload_to_weaviate_incremental.py   # Smart upload with duplicate prevention
├── check_database_status.py            # Database status and verification tool
├── analyze_database_size.py            # Comprehensive database size analysis
├── processed_files_tracking.json       # File tracking with MD5 hashes
├── docker-compose.yml                  # Weaviate database setup with Docker
├── requirements.txt                    # Python dependencies (updated)
├── .env.example                        # Environment template
└── abstracts/                         # Directory for PDF files (84+ medical abstracts)
```

## 🐳 Weaviate with Docker

### Docker Setup & Management
Weaviate runs as a containerized vector database, providing persistent storage and fast semantic search:

```bash
# Start Weaviate (detached mode)
docker-compose up -d

# Check container status
docker ps
# Should show: emarag-weaviate-weaviate-1

# View container logs
docker-compose logs weaviate

# Stop Weaviate
docker-compose down

# Stop and remove all data (⚠️ destructive)
docker-compose down -v

# Restart Weaviate service
docker-compose restart weaviate
```

### Weaviate Configuration
The `docker-compose.yml` configures Weaviate with:
- **Port**: 8080 (accessible at http://localhost:8080)
- **Persistence**: Docker volume for data retention
- **Memory**: Optimized for vector operations
- **Modules**: Text vectorization and search capabilities

### Database Size Monitoring
```bash
# Quick size check
docker exec emarag-weaviate-weaviate-1 du -sh /var/lib/weaviate

# Comprehensive analysis
python analyze_database_size.py

# Database content overview
python check_database_status.py

# Docker resource usage
docker stats --no-stream
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# LM Studio Configuration (optional)
LM_STUDIO_BASE_URL=http://192.168.2.64:1234

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
```

### LLM Provider Setup

#### Option 1: LM Studio (Local)
1. Install [LM Studio](https://lmstudio.ai/)
2. Download models:
   - `deepseek/deepseek-r1-0528-qwen3-8b`
   - `google/gemma-3-12b`
3. Start LM Studio server on port 1234

#### Option 2: OpenAI (Cloud)
1. Get API key from [OpenAI](https://platform.openai.com/)
2. Add to `.env` file
3. Select models in interface:
   - `gpt-4.1-nano-2025-04-14`
   - `chatgpt-4o-latest`

## 💡 Usage Examples

### RAG Mode (Emergency Medicine Literature)
```
What is REBOA and how is it used in trauma care?
How do you calculate the Sudbury vertigo score?
What are the indications for packed red blood cell transfusion?
What are the latest findings on sepsis management?
```

### General Knowledge Mode
```
@llm What are the symptoms of diabetes?
@llm Explain the pathophysiology of heart failure
@llm What is the mechanism of action of aspirin?
@llm How does CPR work physiologically?
```

## 🏥 Medical Data

The system currently includes **84 emergency medicine abstracts** covering:
- **Trauma Care**: REBOA, hemorrhage control, resuscitation protocols
- **Cardiology**: Heart failure, arrhythmias, interventions, troponin levels  
- **Pediatrics**: Emergency procedures, medications, safety protocols
- **Diagnostics**: Imaging protocols, laboratory values, clinical decision rules
- **Pharmacology**: Emergency medications, dosing guidelines, contraindications
- **Toxicology**: Overdose management, antidotes, poison control
- **Neurology**: Stroke care, seizure management, cognitive assessment

### Database Stats
- **Total Documents**: 84 medical abstracts
- **Total Sentences**: 3,600+ indexed for retrieval
- **Smart Processing**: Only new PDFs processed on updates
- **Cross-Encoder Reranking**: Improved relevance with sentence-transformers

## 🚀 Smart PDF Pipeline

### Automated Processing
The smart pipeline automatically handles PDF management with efficiency:

```bash
# Process only new/changed PDFs
./smart_pdf_pipeline.sh

# Check database status
python check_database_status.py

# Force complete rebuild (if needed)
./smart_pdf_pipeline.sh --force
```

### Features
- **Incremental Processing**: Only processes new or modified PDFs
- **MD5 Hash Tracking**: Prevents duplicate processing
- **Automated Reporting**: Shows processing statistics
- **Error Recovery**: Graceful handling of failed files
- **Status Verification**: Database content validation

### Adding New Content
1. **Drag & Drop**: Copy PDFs to `/home/tourniquetrules/abstracts/`
2. **Run Pipeline**: Execute `./smart_pdf_pipeline.sh`
3. **Verify**: Check output for processing summary
4. **Test**: New content immediately available in chatbot

## ⚡ Performance Benchmarks

### Typical Performance (RTX 4090)
- **LM Studio (DeepSeek R1 8B)**: ~135 tokens/second
- **LM Studio (Gemma 3 12B)**: ~85 tokens/second  
- **OpenAI GPT-4.1 Nano**: ~25 tokens/second
- **OpenAI GPT-4o Latest**: ~15 tokens/second
- **Cross-Encoder Reranking**: ~50ms for 15 contexts
- **spaCy Entity Processing**: ~10ms per query (GPU) / ~30ms (CPU)

### Query Processing Pipeline
- **Vector Search**: ~50-100ms
- **Cross-Encoder Reranking**: ~50-150ms  
- **Context Retrieval**: ~200-500ms
- **Total RAG Pipeline**: ~1-3 seconds (excluding LLM)

### PDF Processing Performance
- **PDF Extraction**: ~2-5 seconds per document
- **Sentence Segmentation**: ~1-3 seconds per document
- **Vector Embedding**: ~50-100ms per sentence
- **Batch Upload**: ~100-500 sentences per second

## 🛠️ Development

### Adding New Medical Data
```bash
# Simple method: drag PDFs to abstracts folder, then:
./smart_pdf_pipeline.sh

# Manual method:
python process_abstracts_incremental.py
python upload_to_weaviate_incremental.py
```

### Customizing the Interface
- Modify `medical_rag_chatbot.py` for UI changes
- Update model configurations in global variables  
- Adjust performance tracking in LLM functions
- Add new example queries in the interface

### Database Management
```bash
# Check current status
python check_database_status.py

# Reset Weaviate database
docker-compose down -v
docker-compose up -d

# Re-upload all data
./smart_pdf_pipeline.sh --force
```

## 🐛 Troubleshooting

### Common Issues

**1. Weaviate Connection Failed**
```bash
# Check if Docker is running
docker ps

# Restart Weaviate
docker-compose restart

# Check logs
docker-compose logs weaviate
```

**2. spaCy Model Missing or GPU Error**
```bash
# Download model
python -m spacy download en_core_sci_scibert

# Test GPU fallback (should work on CPU)
python -c "import spacy; nlp = spacy.load('en_core_sci_scibert'); print('✅ spaCy working')"
```

**3. Cross-Encoder Issues**
```bash
# Install/reinstall sentence-transformers
pip install --upgrade sentence-transformers torch

# Test cross-encoder
python -c "from sentence_transformers import CrossEncoder; print('✅ Cross-encoder available')"
```

**4. LM Studio Connection Error**
- Ensure LM Studio is running on the correct port (1234)
- Check firewall settings
- Verify model is loaded and server is started in LM Studio
- Test connection: `curl http://192.168.2.64:1234/v1/models`

**5. Smart Pipeline Errors**
```bash
# Check file permissions
ls -la /home/tourniquetrules/abstracts/

# Fix permissions if needed
sudo chown -R $USER:$USER /home/tourniquetrules/abstracts/

# Check tracking file
cat processed_files_tracking.json | jq length
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/emarag-weaviate.git
cd emarag-weaviate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Weaviate**: Vector database infrastructure and semantic search
- **spaCy & SciSpaCy**: Medical NLP capabilities and named entity recognition
- **Sentence Transformers**: Cross-encoder reranking for improved relevance
- **Gradio**: Modern web interface framework
- **Emergency Medicine Community**: For providing valuable medical literature
- **Hugging Face**: Transformer models and infrastructure

## 📧 Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review logs in the terminal output
1. Check the [Issues](https://github.com/yourusername/emarag-weaviate/issues) page
2. Create a new issue with detailed information
3. Include system specs and error logs

---

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. Always consult qualified medical professionals for clinical decisions.
