#!/usr/bin/env python3
"""
Simple test script to verify the Emergency Medicine RAG Chatbot functionality.
This script tests basic components without starting the full web interface.
"""

import sys
import os
import time
from typing import List, Dict

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required libraries can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import weaviate
        print("✅ Weaviate client imported successfully")
    except ImportError as e:
        print(f"❌ Weaviate import failed: {e}")
        return False
    
    try:
        import spacy
        print("✅ spaCy imported successfully")
    except ImportError as e:
        print(f"❌ spaCy import failed: {e}")
        return False
    
    try:
        import gradio
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    return True

def test_spacy_model():
    """Test that the spaCy medical model can be loaded."""
    print("\n🧠 Testing spaCy medical model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_sci_scibert")
        print("✅ spaCy SciSciBERT model loaded successfully")
        
        # Test processing
        doc = nlp("The patient has acute myocardial infarction.")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"✅ Entity extraction test: {entities}")
        
        return True
    except OSError:
        print("❌ spaCy SciSciBERT model not found")
        print("💡 Run: python -m spacy download en_core_sci_scibert")
        return False
    except Exception as e:
        print(f"❌ spaCy model test failed: {e}")
        return False

def test_weaviate_connection():
    """Test connection to Weaviate database."""
    print("\n🗄️ Testing Weaviate connection...")
    
    try:
        import weaviate
        client = weaviate.connect_to_local(skip_init_checks=True)
        
        # Test connection
        if client.is_ready():
            print("✅ Weaviate connection successful")
            
            # Test schema
            collections = client.collections.list_all()
            print(f"✅ Found {len(collections)} collections")
            
            client.close()
            return True
        else:
            print("❌ Weaviate is not ready")
            return False
            
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        print("💡 Make sure Weaviate is running: docker-compose up -d")
        return False

def test_environment():
    """Test environment configuration."""
    print("\n⚙️ Testing environment configuration...")
    
    # Check for .env file
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found (optional)")
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️ OPENAI_API_KEY not set (OpenAI features disabled)")
    
    lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.2.64:1234")
    print(f"✅ LM Studio URL: {lm_studio_url}")
    
    return True

def test_medical_rag_components():
    """Test core medical RAG components."""
    print("\n🏥 Testing medical RAG components...")
    
    try:
        # Import the main module components
        from medical_rag_chatbot import (
            initialize_spacy, 
            initialize_weaviate,
            process_query_with_spacy
        )
        
        # Test spaCy initialization
        print("🧠 Testing spaCy initialization...")
        initialize_spacy()
        
        # Test Weaviate initialization
        print("🗄️ Testing Weaviate initialization...")
        weaviate_connected = initialize_weaviate()
        
        if weaviate_connected:
            print("✅ Weaviate initialization successful")
            
            # Test query processing
            print("🔍 Testing query processing...")
            enhanced_query = process_query_with_spacy("What is myocardial infarction?")
            print(f"✅ Query enhancement: '{enhanced_query}'")
            
        return weaviate_connected
        
    except Exception as e:
        print(f"❌ Medical RAG component test failed: {e}")
        return False

def run_performance_test():
    """Run basic performance test."""
    print("\n⚡ Running performance test...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_sci_scibert")
        
        # Test processing speed
        test_text = "The patient presents with acute chest pain, shortness of breath, and elevated troponin levels suggesting myocardial infarction."
        
        start_time = time.time()
        doc = nlp(test_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"✅ NLP processing time: {processing_time:.3f} seconds")
        print(f"✅ Entities found: {len(entities)}")
        
        if processing_time < 1.0:
            print("✅ Performance: Good")
        elif processing_time < 3.0:
            print("⚠️ Performance: Acceptable")
        else:
            print("❌ Performance: Slow (consider GPU acceleration)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏥 Emergency Medicine RAG Chatbot - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("spaCy Model Test", test_spacy_model),
        ("Environment Test", test_environment),
        ("Weaviate Connection Test", test_weaviate_connection),
        ("Medical RAG Components Test", test_medical_rag_components),
        ("Performance Test", run_performance_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
