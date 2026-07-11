from google import genai

from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class GeminiEmbeddings(Embeddings):

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts):

        response = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=texts
        )

        return [embedding.values for embedding in response.embeddings]

    def embed_query(self, text):

        response = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )

        return response.embeddings[0].values


def create_vector_store(pdf_path, api_key):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embedding_model = GeminiEmbeddings(api_key)

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


def get_answer(vector_store, question, api_key):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know."

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using the provided context.

If the answer is partially present,
answer only what is supported by the context.

If the answer cannot be found,
reply exactly:

I don't know.

Do not make up facts.

Context:
{context}

Question:
{question}
"""

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"
