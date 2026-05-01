from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# Load PDF
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


# Split text
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# Embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Vector DB
def create_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


# Search relevant chunks
def search_docs(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever.invoke(query)


# LLM
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key="your api key"
    )

# Ask question
def ask_question(query, docs, llm):
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based only on the context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content


# MAIN
if __name__ == "__main__":
    docs = load_pdf("C-34 Assignment 2.pdf")
    chunks = split_docs(docs)

    embeddings = get_embeddings()
    vectorstore = create_vector_store(chunks, embeddings)

    query = input("give info about the pdf: ")

    relevant_docs = search_docs(query, vectorstore)

    llm = get_llm()
    answer = ask_question(query, relevant_docs, llm)

    print("\nANSWER:\n", answer)
