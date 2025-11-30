from src.rewriting.apply_method import apply_rewriting_method

# Test with different questions to trigger all modes
questions = [
    "Why do I fear abandonment?",
    "Why do I always repeat toxic relationships?",
    "What is attachment anxiety?",
    "Why do I feel empty?"
]

for q in questions:
    print("\n--- Testing:", q)
    result = apply_rewriting_method(q)
    print(result)
