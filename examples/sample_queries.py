# Example Medical Queries for Testing

# Emergency Medicine RAG Queries (use medical literature)
rag_queries = [
    "What is REBOA and how is it used in trauma care?",
    "How do you calculate the Sudbury vertigo score?",
    "What are the indications for packed red blood cell transfusion?",
    "When should antibiotics be used for epistaxis?",
    "What is the Glasgow Coma Scale?",
    "How is sepsis diagnosed in the emergency department?",
    "What are the contraindications for thrombolytic therapy?",
    "How do you manage cardiac arrest in pregnancy?",
    "What is the rapid sequence intubation protocol?",
    "When is thoracotomy indicated in trauma?",
]

# General Medical Knowledge Queries (use @llm prefix)
general_queries = [
    "@llm What are the symptoms of diabetes?",
    "@llm Explain the pathophysiology of heart failure",
    "@llm What is the mechanism of action of aspirin?",
    "@llm How does the complement system work?",
    "@llm What are the stages of wound healing?",
    "@llm Describe the anatomy of the heart",
    "@llm What is the difference between Type 1 and Type 2 diabetes?",
    "@llm How do beta-blockers work?",
    "@llm What is the blood-brain barrier?",
    "@llm Explain the process of inflammation",
]

# Performance Testing Queries
performance_queries = [
    "Quick query for response time testing",
    "@llm What is hypertension?",
    "What is trauma-induced coagulopathy?",
    "@llm Define shock",
]

# Model Comparison Queries (good for testing both LLM providers)
comparison_queries = [
    "Explain acute myocardial infarction management",
    "@llm What is pneumonia?",
    "How is stroke diagnosed in the emergency department?",
    "@llm What are the types of anemia?",
]
