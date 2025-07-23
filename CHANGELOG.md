# Changelog

All notable changes to the Emergency Medicine RAG Chatbot project will be documented in this file.

## [1.0.0] - 2025-07-23

### Added
- 🏥 **Emergency Medicine RAG Chatbot** with Weaviate vector database
- 🤖 **Dual LLM Support**: LM Studio (local) and OpenAI (cloud) integration
- 🧠 **Query Modes**: RAG mode (default) and General Knowledge mode (`@llm` prefix)
- 🔍 **Advanced RAG Pipeline** with spaCy SciSciBERT medical NLP
- ⚡ **Performance Monitoring** with real-time tokens/second tracking
- 🎨 **Modern Gradio Web Interface** with model selection and status display
- 📚 **Medical Literature Integration** with emergency medicine abstracts
- 🐋 **Docker Compose** setup for Weaviate database
- 🎯 **Configurable Response Length** with 1K-4K token slider

### Features
- **LM Studio Models**: DeepSeek R1 8B, Gemma 3 12B
- **OpenAI Models**: GPT-4.1 Nano, GPT-4o Latest
- **GPU Acceleration**: CUDA support for spaCy processing
- **Smart Query Enhancement**: Entity extraction and optimization
- **Relevance Scoring**: Context ranking with citation sources
- **Live Status Updates**: Model selection and performance metrics
- **Interactive Examples**: Built-in query suggestions
- **Automatic Citations**: Source attribution for all RAG responses

### Technical Stack
- **Vector Database**: Weaviate 1.25+ with Docker
- **NLP Framework**: spaCy with SciSciBERT medical model
- **Web Framework**: Gradio 4.0+ for modern UI
- **LLM Integration**: OpenAI API and LM Studio local server
- **Data Processing**: PDF parsing with pdfplumber and pypdf
- **Environment**: Python 3.12+ with virtual environment

### Performance
- **LM Studio**: ~135 tokens/second (DeepSeek R1 8B)
- **OpenAI Nano**: ~25 tokens/second optimized model
- **Vector Search**: ~50-100ms query processing
- **Total Pipeline**: ~1-3 seconds (excluding LLM generation)

### Documentation
- Comprehensive README.md with setup instructions
- Example queries and usage patterns
- Troubleshooting guide
- Performance benchmarks
- Medical disclaimer and safety notes

### Security & Safety
- Environment variable configuration
- API key protection
- Medical disclaimer for educational use
- Error handling and graceful degradation

---

## Development Notes

### Architecture Decisions
1. **Weaviate vs. Other Vector DBs**: Chosen for Docker support and schema flexibility
2. **spaCy SciSciBERT**: Medical domain-specific NLP for better query understanding
3. **Gradio Interface**: Rapid prototyping with professional appearance
4. **Dual LLM Support**: Flexibility between local privacy and cloud capability

### Future Enhancements
- [ ] Multiple medical domains (radiology, pathology, etc.)
- [ ] Advanced query analysis and intent detection
- [ ] Model fine-tuning on medical literature
- [ ] Multi-language support
- [ ] Integration with medical databases (PubMed, etc.)
- [ ] Advanced citation and reference management
- [ ] User authentication and session management
- [ ] API endpoints for programmatic access

### Known Limitations
- Requires significant GPU memory for optimal performance
- Medical literature limited to emergency medicine abstracts
- LM Studio requires manual model management
- Single-user interface (no multi-tenancy)

---

*This project is for educational and research purposes only. Always consult qualified medical professionals for clinical decisions.*
