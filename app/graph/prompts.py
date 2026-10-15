from langchain_core.prompts import ChatPromptTemplate

ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You prepare questions for a search over the FastAPI tutorial documentation.\n"
                "1. search_query: rewrite the question as a standalone, keyword-rich search query. "
                "Clarify ambiguity and add likely synonyms or FastAPI terms. Do not answer it.\n"
                "2. query_type, one of:\n"
                "- conceptual: what something is or why it works that way\n"
                "- how-to: steps to achieve something\n"
                "- troubleshooting: an error or unexpected behaviour\n"
                "- api-reference: the parameters, options or signature of a specific function or class"
            ),
        ),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You judge whether a documentation excerpt helps answer a question. "
                "Say relevant only if the excerpt contains information that helps answer "
                "the question. Sharing a topic or keyword is not enough."
            ),
        ),
        ("human", "Question: {question}\n\nExcerpt:\n{context}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "A search query returned nothing useful from the FastAPI documentation. "
                "Write a different query for the same question: use other terms or synonyms, "
                "or make it broader or narrower. Never repeat the failed query."
            ),
        ),
        ("human", "Question: {question}\nFailed query: {failed_query}"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer the question using only the documentation excerpts provided. "
                "Each excerpt starts with its source in the form [source: file.md | section].\n"
                "- Cite the excerpts you use inline as [source: file.md], using the exact file name.\n"
                "- Keep code examples exactly as they appear in the excerpts.\n"
                "- If the excerpts only partly cover the question, answer the part they cover "
                "and say what is missing. Do not guess or use outside knowledge."
            ),
        ),
        ("human", "Question: {question}\n\nDocumentation excerpts:\n{context}"),
    ]
)
