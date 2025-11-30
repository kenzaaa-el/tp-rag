from src.router.router import query_router
from src.rewriting.multi_query import multi_query_rewriting
from src.rewriting.hyde import hyde_generation
from src.rewriting.self_ask import self_ask_decomposition



def apply_rewriting_method(query: str):
    routing = query_router(query)
    rtype = routing["type"]

    print(f"[Router] Selected method: {rtype}")

    if rtype == "multi_query":
        rewrites = multi_query_rewriting(query)
        return {
            "mode": "multi_query",
            "queries": rewrites
        }

    elif rtype == "hyde":
        synthetic_doc = hyde_generation(query)
        return {
            "mode": "hyde",
            "document": synthetic_doc
        }

    elif rtype == "self_ask":
        subquestions = self_ask_decomposition(query)
        return {
            "mode": "self_ask",
            "queries": subquestions
        }

    else:
        return {
            "mode": "none",
            "query": query
        }
