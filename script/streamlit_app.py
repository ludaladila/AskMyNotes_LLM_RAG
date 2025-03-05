import os
import streamlit as st
from dotenv import load_dotenv

from embedding import get_embedding
from vector import VectorStore
from parse import PDFTextExtractor
from chunk import SimpleTextChunker
from llm import ask_llm

# Load env
load_dotenv()

# Init vector store (connect to existing Pinecone index)
if "store" not in st.session_state:
    st.session_state["store"] = VectorStore()

st.title("📚 RAG Note Assistant - Upload & Ask")

PDF_FOLDER = "pdf_folder"
os.makedirs(PDF_FOLDER, exist_ok=True)

# PDF uploader (only used to trigger processing)
uploaded_files = st.file_uploader("Upload new PDF documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        extractor = PDFTextExtractor(PDF_FOLDER)
        document = extractor.extract_text_from_pdf(file_path)

        chunker = SimpleTextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.process_document(document)

        embeddings = [get_embedding(chunk["content"]) for chunk in chunks]
        st.session_state["store"].add(embeddings, chunks)

        st.success(f"✅ '{file.name}' added to Pinecone!")

# Question box
question = st.text_input("Enter your question")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        query_embedding = get_embedding(question)
        relevant_chunks = st.session_state["store"].search(query_embedding)

        if not relevant_chunks:
            st.warning("No relevant content found, please upload PDFs.")
        else:
            context = "\n".join([chunk["text"] for chunk in relevant_chunks])

            with st.spinner('AI is thinking...'):
                answer = ask_llm(question, context)

            st.markdown("### 🤖 AI Answer")
            st.write(answer)

            st.markdown("### 📖 Reference Chunks")
            st.write(context)
