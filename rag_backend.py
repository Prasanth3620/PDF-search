from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ( GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI,)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def create_vector_store(pdf_path, api_key):
    """
    Creates a FAISS vector database from the uploaded PDF.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


def get_answer(vector_store, question, api_key):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":3}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY using the given context.

If the answer is not present in the context,
reply with:

"I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context","question"]
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.2,
        google_api_key=api_key
    )

    response = llm.invoke(final_prompt)

    # Gemini sometimes returns a list of text parts
    if isinstance(response.content, list):

        return "".join(
            part["text"]
            for part in response.content
            if part["type"] == "text"
        )

    return response.content
