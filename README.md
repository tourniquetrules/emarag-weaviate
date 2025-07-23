# Emergency Medicine RAG Chatbot with Weaviate

A sophisticated Retrieval-Augmented Generation (RAG) chatbot specialized in emergency medicine, powered by Weaviate vector database and supporting multiple LLM providers.

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
./start_chatbot.sh

# Visit http://localhost:7871
```

## 🌟 Features

### 🤖 Dual LLM Support
- **LM Studio Integration**: Local models (DeepSeek R1 8B, Gemma 3 12B)
- **OpenAI Integration**: Cloud models (GPT-4.1 Nano, GPT-4o Latest)
- **Model Selection**: Choose specific models through dropdown interface

### 🧠 Query Modes
- **📚 RAG Mode (Default)**: Answers based on emergency medicine literature
- **🚀 General Knowledge Mode**: Use `@llm` prefix for general medical knowledge

### 🔍 Advanced RAG Pipeline
- **Weaviate Vector Database**: Persistent storage with Docker
- **spaCy SciSciBERT**: Medical NLP with GPU acceleration
- **Smart Query Enhancement**: Entity extraction and query optimization
- **Relevance Scoring**: Context ranking with citation sources

### ⚡ Performance Monitoring
- **Real-time Metrics**: Tokens/second tracking for both providers
- **Configurable Responses**: 1K-4K token slider for response length
- **Performance Comparison**: Side-by-side LLM evaluation

### 🎨 Modern Web Interface
- **Gradio UI**: Clean, responsive web interface
- **Live Status Updates**: Model selection and performance display
- **Interactive Examples**: Built-in query suggestions
- **Citation Sources**: Automatic source attribution

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- NVIDIA GPU (recommended for spaCy acceleration)
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

### 4. Start Weaviate Database
```bash
docker-compose up -d
```

### 5. Upload Medical Data
```bash
# Process and upload emergency medicine abstracts
python upload_to_weaviate.py
```

### 6. Launch Chatbot
```bash
# Using the start script
./start_chatbot.sh

# Or directly
python medical_rag_chatbot.py
```

Visit `http://localhost:7871` to access the web interface.

## 📁 Project Structure

```
emarag-weaviate/
├── medical_rag_chatbot.py      # Main chatbot application
├── upload_to_weaviate.py       # Data ingestion script
├── process_abstracts.py        # Abstract processing utilities
├── sample_query.py             # Example usage script
├── docker-compose.yml          # Weaviate database setup
├── requirements.txt            # Python dependencies
├── start_chatbot.sh           # Launch script
├── .env.example               # Environment template
├── processed_abstracts.jsonl  # Processed medical abstracts
└── processed_abstracts_sentences.jsonl  # Sentence-level data
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

### RAG Mode (Emergency Medicine)
```
What is REBOA and how is it used in trauma care?
How do you calculate the Sudbury vertigo score?
What are the indications for packed red blood cell transfusion?
```

### General Knowledge Mode
```
@llm What are the symptoms of diabetes?
@llm Explain the pathophysiology of heart failure
@llm What is the mechanism of action of aspirin?
```

## 🏥 Medical Data

The system includes emergency medicine abstracts covering:
- **Trauma Care**: REBOA, hemorrhage control, resuscitation
- **Cardiology**: Heart failure, arrhythmias, interventions
- **Pediatrics**: Emergency procedures, medications, protocols
- **Diagnostics**: Imaging, laboratory values, clinical scores
- **Pharmacology**: Emergency medications, dosing, contraindications

## ⚡ Performance Benchmarks

### Typical Performance (RTX 4090)
- **LM Studio (DeepSeek R1 8B)**: ~135 tokens/second
- **LM Studio (Gemma 3 12B)**: ~85 tokens/second
- **OpenAI GPT-4.1 Nano**: ~25 tokens/second
- **OpenAI GPT-4o Latest**: ~15 tokens/second

### Query Processing
- **Vector Search**: ~50-100ms
- **Context Retrieval**: ~200-500ms
- **Total RAG Pipeline**: ~1-3 seconds (excluding LLM)

## 🛠️ Development

### Adding New Medical Data
1. Place PDFs in `abstracts/` directory
2. Run `python process_abstracts.py`
3. Upload with `python upload_to_weaviate.py`

### Customizing the Interface
- Modify `medical_rag_chatbot.py` for UI changes
- Update model configurations in global variables
- Adjust performance tracking in LLM functions

### Database Management
```bash
# Reset Weaviate database
docker-compose down -v
docker-compose up -d

# Re-upload data
python upload_to_weaviate.py
```

## 🐛 Troubleshooting

### Common Issues

**1. Weaviate Connection Failed**
```bash
# Check if Docker is running
docker ps

# Restart Weaviate
docker-compose restart
```

**2. spaCy Model Missing**
```bash
python -m spacy download en_core_sci_scibert
```

**3. LM Studio Connection Error**
- Ensure LM Studio is running on the correct port
- Check firewall settings
- Verify model is loaded in LM Studio

**4. GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA support for spaCy
pip install spacy[cuda-autodetect]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Weaviate**: Vector database infrastructure
- **spaCy**: Medical NLP capabilities
- **Gradio**: Web interface framework
- **Emergency Medicine Community**: For providing valuable medical literature

## 📧 Support

For questions or issues:
1. Check the [Issues](https://github.com/yourusername/emarag-weaviate/issues) page
2. Create a new issue with detailed information
3. Include system specs and error logs

---

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. Always consult qualified medical professionals for clinical decisions.
