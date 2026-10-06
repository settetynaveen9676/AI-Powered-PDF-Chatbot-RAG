import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Powered PDF Chatbot")
st.write(
    "Upload a PDF and ask questions about its content using "
    "Retrieval-Augmented Generation (RAG)."
)

# Check API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.warning(
        "GOOGLE_API_KEY is not configured. "
        "Create a .env file and add your Google Gemini API key."
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    st.success(
        f"PDF loaded successfully — {len(documents)} pages found."
    )

    # Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    st.info(f"Created {len(chunks)} text chunks.")

    # Create embeddings
    with st.spinner("Creating document embeddings..."):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success("Document indexed successfully!")

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.2
    )

    # Create retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}
        ),
        return_source_documents=True
    )

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        with st.spinner("Generating answer..."):

            result = qa_chain.invoke(
                {"query": question}
            )

        st.subheader("Answer")

        st.write(result["result"])

        st.subheader("Sources")

        sources = result.get(
            "source_documents",
            []
        )

        for source in sources:

            page = source.metadata.get(
                "page",
                "Unknown"
            )

            st.write(
                f"📄 Page {page + 1 if isinstance(page, int) else page}"
            )
