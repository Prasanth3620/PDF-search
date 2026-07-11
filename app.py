import streamlit as st
import tempfile

from rag_backend import create_vector_store, get_answer


st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄"
)

st.title("📄 PDF Question Answering")

st.write("Upload a PDF and ask questions about it.")

API_KEY = st.secrets["GOOGLE_API_KEY"]

# st.write(st.secrets)
# st.stop()

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# Create vector database only once
if uploaded_pdf is not None:

    if (
        "pdf_name" not in st.session_state
        or st.session_state.pdf_name != uploaded_pdf.name
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_pdf.read())

            pdf_path = tmp.name

        with st.spinner("Reading PDF and creating embeddings..."):

            vector_store = create_vector_store(
                pdf_path,
                API_KEY
            )

            st.session_state.vector_store = vector_store
            st.session_state.pdf_name = uploaded_pdf.name

        st.success("PDF processed successfully!")



question = st.text_input(
    "Ask a question"
)


if st.button("Get Answer"):

    if uploaded_pdf is None:

        st.warning("Please upload a PDF first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            answer = get_answer(
                st.session_state.vector_store,
                question,
                API_KEY
            )

        st.markdown("### Answer")

        st.write(answer)
