from langchain_core.prompts import ChatPromptTemplate

ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You prepare questions for a search over the FastAPI tutorial documentation.\n"
                "The conversation so far is shown before the question; it may be empty.\n"
                "1. standalone_question: the question rewritten so it makes sense on its own. "
                'If it depends on the conversation ("it", "that", "what about ..."), fill in '
                "what it refers to. If it already stands alone, repeat it unchanged.\n"
                "2. search_query: a keyword-rich search query for the standalone question. "
                "Clarify ambiguity and add likely synonyms or FastAPI terms. Do not answer it.\n"
                "3. query_type, one of:\n"
                "- conceptual: what something is or why it works that way\n"
                "- how-to: steps to achieve something\n"
                "- troubleshooting: an error or unexpected behaviour\n"
                "- api-reference: the parameters, options or signature of a specific function or class"
            ),
        ),
        ("human", "Conversation so far:\n{history}\n\nQuestion: {question}"),
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
                "and say what is missing. Do not guess or use outside knowledge.\n"
                "{extra_instructions}"
            ),
        ),
        ("human", "Question: {question}\n\nDocumentation excerpts:\n{context}"),
    ]
)

GROUNDING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You check an answer against the documentation excerpts it was written from. "
                "List every claim in the answer that the excerpts do not support: invented "
                "facts, wrong details, or advice that is not in the excerpts. Code copied from "
                "the excerpts and statements that the excerpts do not cover something count as "
                "supported. Set grounded to true only if there are no unsupported claims."
            ),
        ),
        ("human", "Documentation excerpts:\n{context}\n\nAnswer to check:\n{answer}"),
    ]
)
