# Contributing to Emergency Medicine RAG Chatbot

Thank you for your interest in contributing to the Emergency Medicine RAG Chatbot! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information**:
   - Operating system and version
   - Python version
   - GPU information (if applicable)
   - Error messages and logs
   - Steps to reproduce

### Suggesting Features
1. **Check existing feature requests** first
2. **Describe the problem** you're trying to solve
3. **Explain your proposed solution**
4. **Consider implementation complexity**
5. **Think about medical safety implications**

### Code Contributions

#### Getting Started
1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/emarag-weaviate.git
   cd emarag-weaviate
   ```
3. **Set up development environment**:
   ```bash
   ./setup.sh
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Guidelines

##### Code Style
- **Follow PEP 8** Python style guidelines
- **Use type hints** where appropriate
- **Add docstrings** for functions and classes
- **Keep functions focused** and single-purpose
- **Use descriptive variable names**

##### Medical Safety
- **Medical disclaimer** must be prominent
- **Cite sources** for medical information
- **Validate medical accuracy** before submitting
- **Consider liability implications**
- **Test with medical professionals** when possible

##### Testing
- **Test all LLM providers** (LM Studio, OpenAI)
- **Verify RAG functionality** with sample queries
- **Check performance** on different hardware
- **Test error handling** and edge cases
- **Validate web interface** across browsers

#### File Structure Guidelines
```
emarag-weaviate/
├── medical_rag_chatbot.py      # Main application
├── upload_to_weaviate.py       # Data processing
├── utils/                      # Utility functions
├── tests/                      # Test files
├── docs/                       # Documentation
└── examples/                   # Usage examples
```

#### Commit Guidelines
- **Use conventional commits**: `type(scope): description`
- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Examples**:
  - `feat(llm): add Gemma 3 model support`
  - `fix(weaviate): handle connection timeout`
  - `docs(readme): update installation instructions`

#### Pull Request Process
1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update CHANGELOG.md**
5. **Request review** from maintainers

## 🛠️ Development Setup

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Git
- NVIDIA GPU (recommended)

### Local Development
```bash
# Clone and setup
git clone https://github.com/yourusername/emarag-weaviate.git
cd emarag-weaviate
./setup.sh

# Activate environment
source venv312/bin/activate

# Run in development mode
python medical_rag_chatbot.py
```

### Testing Changes
```bash
# Test basic functionality
python sample_query.py

# Test data upload
python upload_to_weaviate.py

# Test web interface
# Visit http://localhost:7871
```

## 📋 Contribution Areas

### High Priority
- [ ] **Performance optimization** for lower-end hardware
- [ ] **Error handling** improvements
- [ ] **Documentation** enhancements
- [ ] **Medical literature** expansion
- [ ] **User interface** improvements

### Medium Priority
- [ ] **Additional LLM providers** (Anthropic, etc.)
- [ ] **Advanced query features** (filters, categories)
- [ ] **Export functionality** (PDF, citations)
- [ ] **Configuration management** (settings UI)
- [ ] **Logging and monitoring** improvements

### Research Projects
- [ ] **Medical domain expansion** (radiology, pathology)
- [ ] **Multi-language support**
- [ ] **Fine-tuning medical models**
- [ ] **Advanced citation systems**
- [ ] **Integration with medical databases**

## 🔬 Medical Accuracy Guidelines

### Research Standards
- **Use peer-reviewed sources** only
- **Verify information** with multiple sources
- **Include proper citations**
- **Note uncertainty** when appropriate
- **Avoid definitive medical advice**

### Review Process
1. **Medical professional review** (when possible)
2. **Source verification**
3. **Accuracy testing**
4. **Disclaimer compliance**
5. **Legal review** (for significant changes)

## 📚 Resources

### Medical Literature
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [Emergency Medicine journals](https://www.nejm.org/)
- [Clinical practice guidelines](https://www.ahrq.gov/)

### Technical Documentation
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [spaCy Documentation](https://spacy.io/)
- [Gradio Documentation](https://gradio.app/docs/)
- [OpenAI API Reference](https://platform.openai.com/docs/)

### Development Tools
- [Black](https://black.readthedocs.io/) - Code formatting
- [Flake8](https://flake8.pycqa.org/) - Linting
- [MyPy](https://mypy.readthedocs.io/) - Type checking
- [Pytest](https://pytest.org/) - Testing

## ⚖️ Legal Considerations

### Medical Disclaimer
All contributions must maintain the medical disclaimer that this tool is for educational purposes only and should not replace professional medical advice.

### Licensing
- All contributions are subject to the MIT License
- Ensure you have rights to contribute any code or content
- Medical literature must be properly cited and legally obtained

### Privacy
- No patient data should be included
- API keys should use environment variables
- Consider HIPAA implications for medical applications

## 🆘 Getting Help

### Community Support
- **GitHub Issues**: Technical problems and feature requests
- **Discussions**: General questions and ideas
- **Pull Request Reviews**: Code feedback and suggestions

### Documentation
- **README.md**: Getting started guide
- **CHANGELOG.md**: Version history
- **Code comments**: Inline documentation

### Contact
For sensitive issues or security concerns, please contact the maintainers directly rather than creating public issues.

---

**Thank you for contributing to making medical AI tools better and safer! 🏥**
