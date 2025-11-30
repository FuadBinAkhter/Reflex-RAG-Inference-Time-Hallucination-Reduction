import wikipedia

def get_dynamic_context(query: str) -> str:
    """Retrieves ground truth from Wikipedia to ground the RAG process."""
    print(f"Retrieving knowledge for: '{query}'...")
    try:
        results = wikipedia.search(query)
        if not results: 
            return ""
        return wikipedia.summary(results[0], sentences=5, auto_suggest=False)
    except Exception as e:
        print(f"Retrieval Warning: {e}")
        return ""