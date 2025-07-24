#!/bin/bash

echo "🚀 Setting up Cross-Encoder Reranking for Medical RAG Chatbot"
echo "============================================================"

# Install sentence-transformers if not already installed
echo "📦 Installing sentence-transformers..."
pip install sentence-transformers>=2.7.0

# Test cross-encoder availability
echo "🧪 Testing cross-encoder installation..."
python3 -c "
try:
    from sentence_transformers import CrossEncoder
    print('✅ Cross-encoder available')
    
    # Test loading a model
    print('🔄 Testing cross-encoder model loading...')
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print('✅ Cross-encoder model loaded successfully')
    
    # Test a simple prediction
    pairs = [['What is sepsis?', 'Sepsis is a life-threatening condition caused by infection.']]
    score = model.predict(pairs)
    print(f'✅ Cross-encoder prediction test: {score[0]:.3f}')
    
except ImportError as e:
    print(f'❌ Cross-encoder not available: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  Cross-encoder test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Cross-encoder setup complete!"
    echo ""
    echo "✅ Benefits of cross-encoder reranking:"
    echo "   • Improved relevance scoring of retrieved contexts"
    echo "   • Better medical domain understanding"
    echo "   • More accurate source ranking"
    echo ""
    echo "🔧 The chatbot will now automatically use cross-encoder reranking"
    echo "   when you restart the application."
else
    echo ""
    echo "❌ Cross-encoder setup failed!"
    echo "   Please check your Python environment and try again."
    exit 1
fi
