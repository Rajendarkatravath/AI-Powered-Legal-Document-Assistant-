from langchain_groq import ChatGroq

def generate_summary(docs, llm):
    # Generate a summary of the retrieved documents
    summarizer_prompt = (
        "Summarize the following legal documents with respect to the query:\n\n"
        "{docs}"
    )
    prompt = summarizer_prompt.format(docs=docs)
    return llm.complete(prompt)
