import gradio as gr
import weaviate
import requests
import json
import time
import os
import spacy
import signal
import sys
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Cross-encoder imports for reranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    print("✅ Cross-encoder available for reranking")
except ImportError as e:
    CROSS_ENCODER_AVAILABLE = False
    print(f"⚠️  Cross-encoder not available: {e}")
    print("📦 Run: pip install sentence-transformers")

# Load environment variables
load_dotenv()

# Global variables
client = None
nlp = None
cross_encoder = None  # Cross-encoder for reranking
USE_OPENAI = False
LLM_CHOICE = None
SELECTED_LM_STUDIO_MODEL = None
SELECTED_OPENAI_MODEL = None

# API configuration - from original emergency_rag_chatbot.py
# API configuration - from original emergency_rag_chatbot.py
LM_STUDIO_BASE_URL = "http://192.168.2.64:1234"
LM_STUDIO_MODELS = {
    "DeepSeek R1 8B": "deepseek/deepseek-r1-0528-qwen3-8b",
    "Gemma 3 12B": "google/gemma-3-12b"
}
OPENAI_MODELS = {
    "GPT-4.1 Nano": "gpt-4.1-nano-2025-04-14",
    "GPT-4o Latest": "chatgpt-4o-latest"
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize_spacy():
    """Initialize spaCy with fallback from GPU to CPU"""
    global nlp
    
    # First try GPU
    try:
        spacy.require_gpu()
        nlp = spacy.load("en_core_sci_scibert")
        print("✅ spaCy SciSciBERT model loaded with GPU")
        return
    except Exception as e:
        print(f"⚠️ spaCy GPU failed: {e}")
        print("🔄 Falling back to CPU mode...")
    
    # Fallback to CPU
    try:
        # Ensure we're not trying to use GPU
        spacy.prefer_gpu(0)  # This will fail safely if no GPU
        nlp = spacy.load("en_core_sci_scibert")
        print("✅ spaCy SciSciBERT model loaded with CPU")
    except Exception as e:
        print(f"❌ Failed to load spaCy model: {e}")
        print("🔄 Setting nlp to None - will skip entity processing")
        nlp = None

def initialize_cross_encoder():
    """Initialize cross-encoder for reranking"""
    global cross_encoder
    
    if not CROSS_ENCODER_AVAILABLE:
        print("⚠️  Cross-encoder not available - skipping initialization")
        return False
    
    try:
        # Use a medical/clinical cross-encoder model if available, otherwise general purpose
        models_to_try = [
            'cross-encoder/ms-marco-MiniLM-L-6-v2',  # General purpose, good performance
            'cross-encoder/ms-marco-MiniLM-L-12-v2', # Better quality, slower
        ]
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Loading cross-encoder: {model_name}")
                cross_encoder = CrossEncoder(model_name)
                print(f"✅ Cross-encoder loaded: {model_name}")
                return True
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")
                continue
        
        print("❌ Failed to load any cross-encoder model")
        return False
        
    except Exception as e:
        print(f"❌ Cross-encoder initialization error: {e}")
        return False

def initialize_weaviate():
    """Initialize Weaviate connection"""
    global client
    try:
        client = weaviate.connect_to_local(skip_init_checks=True)
        print("✅ Weaviate connection established")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return False

def query_weaviate(query_text: str, limit: int = 5) -> List[Dict]:
    """Query Weaviate for relevant context with increased limit for reranking"""
    if not client:
        return []
    
    try:
        collection = client.collections.get("AbstractSentence")
        
        # Get more results than needed for reranking (2-3x the final limit)
        search_limit = limit * 3 if cross_encoder else limit
        
        results = collection.query.near_text(
            query=query_text,
            limit=search_limit,
            return_metadata=["score"]
        )
        
        contexts = []
        for obj in results.objects:
            contexts.append({
                "text": obj.properties['sentence'],
                "filename": obj.properties['filename'],
                "page": obj.properties.get('page', 'N/A'),
                "score": obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0
            })
        
        return contexts
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        return []

def rerank_with_cross_encoder(query: str, contexts: List[Dict], final_limit: int = 5) -> List[Dict]:
    """Rerank contexts using cross-encoder for better relevance"""
    global cross_encoder
    
    if not CROSS_ENCODER_AVAILABLE or cross_encoder is None or len(contexts) <= 1:
        print("🔄 Cross-encoder not available, returning original results")
        return contexts[:final_limit]
    
    try:
        print(f"🎯 Reranking {len(contexts)} contexts with cross-encoder...")
        
        # Prepare query-context pairs for cross-encoder
        pairs = [[query, ctx['text']] for ctx in contexts]
        
        # Get cross-encoder scores (higher = more relevant)
        scores = cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to contexts
        for ctx, score in zip(contexts, scores):
            ctx['cross_encoder_score'] = float(score)
            ctx['original_score'] = ctx.get('score', 0.0)
        
        # Sort by cross-encoder score (descending)
        reranked_contexts = sorted(contexts, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        # Take top results after reranking
        final_contexts = reranked_contexts[:final_limit]
        
        print(f"✅ Reranked to top {len(final_contexts)} contexts")
        for i, ctx in enumerate(final_contexts[:3]):  # Show top 3
            print(f"   {i+1}. Cross-encoder: {ctx['cross_encoder_score']:.3f}, Original: {ctx['original_score']:.3f}")
            print(f"      {ctx['text'][:100]}...")
        
        return final_contexts
        
    except Exception as e:
        print(f"⚠️  Cross-encoder reranking failed: {e}")
        return contexts[:final_limit]

def process_query_with_spacy(query: str) -> str:
    """Process user query with spaCy for entity extraction"""
    if not nlp:
        print("⚠️ spaCy not available, using original query")
        return query
    
    try:
        doc = nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Enhance query with extracted entities for better retrieval
        enhanced_query = query
        if entities:
            entity_terms = [ent[0] for ent in entities]
            enhanced_query = f"{query} {' '.join(entity_terms)}"
        
        return enhanced_query
    except Exception as e:
        print(f"⚠️ spaCy processing failed: {e}")
        print("🔄 Using original query without entity enhancement")
        return query

def call_lm_studio(messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Call LM Studio API with performance tracking"""
    import time
    
    try:
        model_name = SELECTED_LM_STUDIO_MODEL or list(LM_STUDIO_MODELS.values())[0]  # Default to first model
        print(f"🔄 Calling LM Studio at {LM_STUDIO_BASE_URL} with model: {model_name}")
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,  # Now configurable
            "stream": False
        }
        
        print(f"📤 Sending request: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        response = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response headers: {response.headers}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"📥 Full response: {json.dumps(response_data, indent=2)}")
            
            # Handle different possible response formats
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    print(f"✅ Extracted content: {content[:200]}...")
                    
                    # Clean up reasoning tags if present (DeepSeek models often include <think> tags)
                    if "<think>" in content and "</think>" in content:
                        think_end = content.find("</think>")
                        if think_end != -1:
                            content = content[think_end + 8:].strip()
                            print(f"🧹 Cleaned content (removed <think> tags): {content[:200]}...")
                    
                    # Calculate performance metrics
                    response_time = end_time - start_time
                    usage = response_data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", len(content.split()) * 1.3)  # Rough estimate
                    tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
                    
                    print(f"⚡ LM Studio Performance: {completion_tokens:.0f} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                    
                    # Return content and performance data
                    return content, {
                        "tokens": completion_tokens,
                        "response_time": response_time,
                        "tokens_per_second": tokens_per_second
                    }
                elif "text" in choice:
                    content = choice["text"]
                    print(f"✅ Extracted text: {content[:200]}...")
                    # Calculate basic performance data
                    response_time = end_time - start_time
                    estimated_tokens = len(content.split()) * 1.3
                    tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0
                    return content, {
                        "tokens": estimated_tokens,
                        "response_time": response_time,
                        "tokens_per_second": tokens_per_second
                    }
            
            # Fallback: try to extract any text content
            if "content" in response_data:
                content = response_data["content"]
                response_time = end_time - start_time
                estimated_tokens = len(content.split()) * 1.3
                tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0
                return content, {
                    "tokens": estimated_tokens,
                    "response_time": response_time,
                    "tokens_per_second": tokens_per_second
                }
            
            error_msg = f"❌ Unexpected response format: {response_data}"
            return error_msg, {"tokens": 0, "response_time": 0, "tokens_per_second": 0}
        else:
            error_msg = f"❌ LM Studio Error: {response.status_code} - {response.text}"
            print(error_msg)
            return error_msg, {"tokens": 0, "response_time": 0, "tokens_per_second": 0}
            
    except requests.exceptions.Timeout:
        return "❌ LM Studio timeout error (request took too long)", {"tokens": 0, "response_time": 0, "tokens_per_second": 0}
    except requests.exceptions.ConnectionError:
        return f"❌ LM Studio connection error: Cannot connect to {LM_STUDIO_BASE_URL}", {"tokens": 0, "response_time": 0, "tokens_per_second": 0}
    except Exception as e:
        error_msg = f"❌ LM Studio error: {str(e)}"
        return error_msg, {"tokens": 0, "response_time": 0, "tokens_per_second": 0}
        print(error_msg)
        return error_msg

def call_openai(messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Call OpenAI API with enhanced formatting and performance tracking"""
    import time
    
    try:
        import openai
        
        # Enhance the system message for better formatting
        enhanced_messages = []
        for msg in messages:
            if msg["role"] == "system":
                enhanced_content = msg["content"] + """
                
Please format your response with:
- **Bold text** for key medical terms, diagnoses, and important points
- Use bullet points or numbered lists when appropriate
- Structure your answer clearly with headings if needed
- Make the direct answer to the question stand out with bold formatting"""
                enhanced_messages.append({"role": "system", "content": enhanced_content})
            else:
                enhanced_messages.append(msg)
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        model_name = SELECTED_OPENAI_MODEL or list(OPENAI_MODELS.values())[0]  # Default to first model
        print(f"🔄 Calling OpenAI with model: {model_name}")
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,  # Using the selected model
            messages=enhanced_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        end_time = time.time()
        
        raw_response = response.choices[0].message.content
        
        # Calculate performance metrics
        response_time = end_time - start_time
        completion_tokens = response.usage.completion_tokens if response.usage else len(raw_response.split()) * 1.3
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        tokens_per_second = completion_tokens / response_time if response_time > 0 else 0
        
        print(f"⚡ OpenAI Performance: {completion_tokens:.0f} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        print(f"📊 OpenAI Usage: Prompt={prompt_tokens} + Completion={completion_tokens:.0f} = {prompt_tokens + completion_tokens:.0f} total tokens")
        print(f"📝 OpenAI Response length: {len(raw_response)} characters")
        
        # Post-process the response for better formatting
        formatted_response = format_medical_response(raw_response)
        
        return formatted_response, {
            "tokens": completion_tokens,
            "response_time": response_time,
            "tokens_per_second": tokens_per_second
        }
        
    except Exception as e:
        return f"❌ OpenAI error: {str(e)}", {"tokens": 0, "response_time": 0, "tokens_per_second": 0}

def format_medical_response(response: str) -> str:
    """Format medical response with better structure and bold key terms"""
    import re
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        # Bold medical terms and key phrases
        medical_terms = [
            r'\b(diagnosis|treatment|management|prognosis|etiology|pathophysiology)\b',
            r'\b(acute|chronic|emergency|urgent|critical)\b',
            r'\b(contraindicated|indicated|recommended|not recommended)\b',
            r'\b(risk factors?|complications?|symptoms?|signs?)\b',
            r'\b(antibiotics?|medications?|drugs?|therapy|therapies)\b',
            r'\b(patients?|epistaxis|sepsis|infection|prophylaxis)\b'
        ]
        
        formatted_para = paragraph
        
        # Apply bold formatting to medical terms (case insensitive)
        for pattern in medical_terms:
            formatted_para = re.sub(pattern, r'**\1**', formatted_para, flags=re.IGNORECASE)
        
        # Bold sentences that start with common medical conclusions
        conclusion_patterns = [
            r'^(Based on.*?)\.',
            r'^(The findings suggest.*?)\.',
            r'^(In conclusion.*?)\.',
            r'^(Therefore.*?)\.',
            r'^(The evidence shows.*?)\.',
            r'^(Studies indicate.*?)\.'
        ]
        
        for pattern in conclusion_patterns:
            formatted_para = re.sub(pattern, r'**\1**\.', formatted_para, flags=re.IGNORECASE)
        
        formatted_paragraphs.append(formatted_para)
    
    # Join paragraphs with proper spacing
    return '\n\n'.join(formatted_paragraphs)

def generate_response(query: str, history: List[Tuple[str, str]], max_tokens: int = 1000) -> Tuple[str, List[Tuple[str, str]], str]:
    """Generate response using RAG pipeline with configurable token limit and performance tracking"""
    print(f"🔍 Processing query: {query}")
    print(f"🔗 LLM Choice: {LLM_CHOICE}, USE_OPENAI: {USE_OPENAI}")
    print(f"🎯 Max tokens: {max_tokens}")
    
    if not client:
        response = "❌ Weaviate is not connected. Please check your Weaviate instance."
        history.append((query, response))
        return "", history, "❌ No connection"
    
    # Check if query starts with @llm for general knowledge bypass
    bypass_rag = query.lower().startswith("@llm")
    if bypass_rag:
        # Remove @llm prefix and any leading whitespace
        clean_query = query[4:].strip()
        print(f"🚀 RAG BYPASS MODE: Using general knowledge for: {clean_query}")
        
        # Create direct LLM messages without RAG context
        system_message = {
            "role": "system",
            "content": """You are an expert medical AI assistant with extensive knowledge in emergency medicine and general healthcare. 
            Answer the question using your general knowledge and medical training. Provide accurate, evidence-based information 
            while being clear about the limitations of your knowledge. If you're uncertain about something, mention it."""
        }
        
        user_message = {
            "role": "user",
            "content": f"Question: {clean_query}"
        }
        
        messages = [system_message, user_message]
        print(f"💬 Prepared {len(messages)} messages for direct LLM (no RAG)")
        
        # Generate response using selected LLM
        if USE_OPENAI:
            print("🌐 Calling OpenAI (General Knowledge)...")
            response, perf_data = call_openai(messages, max_tokens=max_tokens)
        else:
            print("🏠 Calling LM Studio (General Knowledge)...")
            response, perf_data = call_lm_studio(messages, max_tokens=max_tokens)
        
        # Use actual performance data from LLM call
        tokens = perf_data["tokens"]
        response_time = perf_data["response_time"]
        tokens_per_second = perf_data["tokens_per_second"]
        
        performance_info = f"⚡ {LLM_CHOICE} (General Knowledge): {tokens:.0f} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)"
        
        # Add indicator that this was general knowledge mode
        final_response = f"🧠 **General Knowledge Mode** (RAG bypassed)\n\n{response}"
        history.append((query, final_response))
        
        return "", history, performance_info
    
    # Regular RAG pipeline for non-@llm queries
    # Process query with spaCy
    enhanced_query = process_query_with_spacy(query)
    print(f"🧠 Enhanced query: {enhanced_query}")
    
    # Retrieve relevant context from Weaviate
    contexts = query_weaviate(enhanced_query, limit=5)
    print(f"📚 Found {len(contexts)} initial contexts from Weaviate")
    
    if not contexts:
        response = "❌ No relevant information found in the knowledge base."
        history.append((query, response))
        return "", history, "❌ No contexts found"
    
    # Apply cross-encoder reranking if available
    if cross_encoder and len(contexts) > 1:
        contexts = rerank_with_cross_encoder(query, contexts, final_limit=5)
        print(f"🎯 After reranking: {len(contexts)} contexts selected")
    else:
        contexts = contexts[:5]  # Limit to 5 if no reranking
    
    # Build context string
    context_text = "\\n\\n".join([
        f"[Source: {ctx['filename']}, Page: {ctx['page']}, Score: {ctx['score']:.3f}]\\n{ctx['text']}"
        for ctx in contexts
    ])
    
    print(f"📝 Context length: {len(context_text)} characters")
    
    # Create messages for LLM
    system_message = {
        "role": "system",
        "content": """You are an expert medical AI assistant specializing in emergency medicine. 
        Use the provided context from medical literature to answer questions accurately and concisely.
        Always cite your sources when possible. If the context doesn't contain enough information 
        to answer the question, say so clearly."""
    }
    
    user_message = {
        "role": "user",
        "content": f"""Question: {query}

Context from medical literature:
{context_text}

Please provide a comprehensive answer based on the context above."""
    }
    
    messages = [system_message, user_message]
    print(f"💬 Prepared {len(messages)} messages for LLM")
    
    # Generate response using selected LLM
    if USE_OPENAI:
        print("🌐 Calling OpenAI...")
        response, perf_data = call_openai(messages, max_tokens=max_tokens)
    else:
        print("🏠 Calling LM Studio...")
        response, perf_data = call_lm_studio(messages, max_tokens=max_tokens)
    
    # Use actual performance data from LLM call
    tokens = perf_data["tokens"]
    response_time = perf_data["response_time"]
    tokens_per_second = perf_data["tokens_per_second"]
    
    performance_info = f"⚡ {LLM_CHOICE}: {tokens:.0f} tokens in {response_time:.2f}s ({tokens_per_second:.1f} tok/s)"
    
    print(f"🤖 LLM Response length: {len(response)} characters")
    print(f"🤖 LLM Response preview: {response[:200]}...")
    
    # Add sources
    sources = "\\n\\n**Sources:**\\n" + "\\n".join([
        f"• {ctx['filename']} (Page {ctx['page']}, Relevance: {ctx.get('cross_encoder_score', ctx['score']):.3f})"
        for ctx in contexts
    ])
    
    final_response = response + sources
    history.append((query, final_response))
    
    return "", history, performance_info

def setup_llm_choice():
    """Setup LLM choice interface"""
    global USE_OPENAI, LLM_CHOICE, SELECTED_LM_STUDIO_MODEL, SELECTED_OPENAI_MODEL
    
    def set_openai(model_selection):
        global USE_OPENAI, LLM_CHOICE, SELECTED_OPENAI_MODEL
        USE_OPENAI = True
        LLM_CHOICE = "OpenAI"
        SELECTED_OPENAI_MODEL = OPENAI_MODELS[model_selection]
        return f"✅ Using OpenAI: {model_selection} ({SELECTED_OPENAI_MODEL})"
    
    def set_lm_studio(model_selection):
        global USE_OPENAI, LLM_CHOICE, SELECTED_LM_STUDIO_MODEL
        USE_OPENAI = False
        LLM_CHOICE = "LM Studio"
        SELECTED_LM_STUDIO_MODEL = LM_STUDIO_MODELS[model_selection]
        return f"✅ Using LM Studio: {model_selection} ({SELECTED_LM_STUDIO_MODEL})"

    return set_openai, set_lm_studio

def create_interface():
    """Create Gradio interface"""
    # Initialize components
    initialize_spacy()
    cross_encoder_loaded = initialize_cross_encoder()
    weaviate_connected = initialize_weaviate()
    
    if not weaviate_connected:
        gr.Warning("⚠️ Weaviate connection failed. Check if Weaviate is running.")
    
    set_openai, set_lm_studio = setup_llm_choice()
    
    with gr.Blocks(title="Emergency Medicine RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1>🏥 Emergency Medicine RAG Chatbot</h1>")
        
        # Add prominent note about @llm feature
        gr.HTML("""
        <div style="background-color: #f0f8ff; border: 3px solid #0066cc; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; margin-bottom: 15px; color: #0066cc; font-weight: bold; font-size: 18px;">💡 Query Modes</h3>
            <p style="margin-bottom: 10px; color: #333; font-size: 14px; line-height: 1.4;"><strong style="color: #0066cc;">📚 RAG Mode (Default):</strong> Ask about emergency medicine topics to get answers from medical literature</p>
            <p style="margin-bottom: 10px; color: #333; font-size: 14px; line-height: 1.4;"><strong style="color: #0066cc;">🧠 General Knowledge Mode:</strong> Type <code style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 4px; color: #0066cc; font-weight: bold;">@llm</code> before your question to use general medical knowledge (bypasses RAG)</p>
            <p style="margin-bottom: 0; color: #555; font-style: italic; font-size: 13px;">Example: <code style="background-color: #e6f3ff; padding: 2px 6px; border-radius: 4px; color: #0066cc;">@llm What are the symptoms of diabetes?</code></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🤖 LLM Selection</h3>")
                
                # OpenAI Selection
                gr.HTML("<h4>🌐 OpenAI Models</h4>")
                openai_model_dropdown = gr.Dropdown(
                    choices=list(OPENAI_MODELS.keys()),
                    value=list(OPENAI_MODELS.keys())[0],
                    label="Select OpenAI Model",
                    interactive=True
                )
                openai_btn = gr.Button("Use Selected OpenAI Model", variant="secondary")
                
                # LM Studio Selection
                gr.HTML("<h4>🏠 LM Studio Models</h4>")
                lm_studio_model_dropdown = gr.Dropdown(
                    choices=list(LM_STUDIO_MODELS.keys()),
                    value=list(LM_STUDIO_MODELS.keys())[0],
                    label="Select LM Studio Model",
                    interactive=True
                )
                lm_studio_btn = gr.Button("Use Selected LM Studio Model", variant="secondary")
                
                llm_status = gr.Textbox(
                    label="Current LLM",
                    value="Please select an LLM provider",
                    interactive=False
                )
                
                gr.HTML("<h3>⚙️ Response Settings</h3>")
                token_slider = gr.Slider(
                    minimum=1000,
                    maximum=4000,
                    step=1000,
                    value=1000,
                    label="Max Tokens",
                    info="1K for shorter, 4K for detailed responses"
                )
                
                gr.HTML("<h3>📊 System Status</h3>")
                status_text = f"""
                **Weaviate:** {'✅ Connected' if weaviate_connected else '❌ Disconnected'}  
                **spaCy SciSciBERT:** ✅ Loaded  
                **Cross-Encoder Reranking:** {'✅ Enabled' if cross_encoder_loaded else '⚠️ Disabled'}  
                **GPU Acceleration:** {'✅ Enabled' if spacy.prefer_gpu() else '⚠️ CPU Only'}
                """
                gr.Markdown(status_text)
                
                gr.HTML("<h3>⚡ Performance</h3>")
                performance_display = gr.Textbox(
                    label="Last Response Performance",
                    value="No queries yet",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_label=True
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a medical question (or type @llm for general knowledge)",
                        placeholder="RAG: What is REBOA? | General: @llm What is hypertension?",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                gr.HTML("""
                <h4>💡 Example Questions:</h4>
                <ul>
                    <li><strong>RAG Mode:</strong> What is REBOA and how is it used in trauma care?</li>
                    <li><strong>RAG Mode:</strong> How do you calculate the Sudbury vertigo score?</li>
                    <li><strong>RAG Mode:</strong> What are the indications for packed red blood cell transfusion?</li>
                    <li><strong>General Knowledge:</strong> @llm What are the symptoms of diabetes?</li>
                    <li><strong>General Knowledge:</strong> @llm Explain the pathophysiology of heart failure</li>
                </ul>
                <p><strong>💡 Tip:</strong> Use <code>@llm</code> prefix to bypass RAG and use general knowledge!</p>
                """)
        
        # Event handlers
        openai_btn.click(
            set_openai, 
            inputs=[openai_model_dropdown], 
            outputs=[llm_status]
        )
        lm_studio_btn.click(
            set_lm_studio, 
            inputs=[lm_studio_model_dropdown], 
            outputs=[llm_status]
        )
        
        submit_btn.click(
            generate_response,
            inputs=[query_input, chatbot, token_slider],
            outputs=[query_input, chatbot, performance_display]
        )
        
        query_input.submit(
            generate_response,
            inputs=[query_input, chatbot, token_slider],
            outputs=[query_input, chatbot, performance_display]
        )
    
    return demo

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    print("✅ Emergency Medicine RAG Chatbot stopped")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🏥 Starting Emergency Medicine RAG Chatbot...")
    print("🌐 Web interface will be available at: http://localhost:7871")
    print("⏹️  Send SIGTERM or SIGINT to stop gracefully")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7871,  # Changed from 7870 to avoid port conflicts
        share=False,
        show_error=True
    )
