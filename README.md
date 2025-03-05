# AskMyNotes_LLM_RAG
## Objective

Over time, I've collected a large number of lecture notes, technical documents, and study files. Existing note tools like Notion help with organization, but they still make it hard to quickly locate specific knowledge, especially when notes are long and cover multiple topics.
This project aims to build a personal note assistant powered by Retrieval-Augmented Generation (RAG) to solve this problem. By semantically understanding your documents and queries, it provides more relevant answers than traditional keyword search.

##  Features

- **PDF Processing**: Extract text from PDF documents with support for multilingual content 
- **Smart Text Chunking**: Break documents into semantic chunks with configurable size and overlap
- **Vector Storage**: Store document embeddings in Pinecone for efficient similarity search
- **Multilingual Support**: Process and understand content in multiple languages 
- **Interactive UI**: Simple Streamlit interface for uploading documents and asking questions
- **Fast Retrieval**: Quickly find relevant information across all your documents

##  Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   DEEPSEEK_API_KEY=your_deepseek_api_key
   ```

##  Usage
### Running the App on the cloud
The link is [AskmyNote_Link](https://askmynotesllmrag-mtk6ptfpepyaw4hvt5jpcz.streamlit.app/)

### Running the  App (local)

Start the application:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501` to access the interface.

### Using the Application

1. **Ask Questions**: Enter your question in the text box and click "Submit"
2. **View Results**: The system will display the AI's answer along with reference chunks from your documents
3. **Upload Documents**: Use the file uploader to add more PDF files to the knowledge base


## Project Structure

- `app.py` - Streamlit application entry point
- `parse.py` - PDF text extraction utilities
- `chunk.py` - Text chunking algorithms
- `embedding.py` - Text embedding functions
- `vector.py` - Vector database interface
- `pipe.py` - End-to-end pipeline (extraction, chunking, embedding)
- `llm.py` - LLM interface for answering questions

## Module Details

### PDFTextExtractor

Extracts and cleans text from PDF documents:

- Handles multi-page documents
- Cleans and normalizes text
- Preserves paragraph structure
- Supports multilingual content including Chinese

### SimpleTextChunker

Splits documents into semantic chunks:

- Paragraph-aware chunking where possible
- Configurable chunk size and overlap
- Support for recursive chunking based on document structure
- Special handling for Chinese text

### VectorStore

Interface to Pinecone vector database:

- Creates and manages Pinecone indexes
- Stores document chunks with metadata
- Performs similarity search with configurable parameters

##  Workflow

1. **Text Extraction**: PDF documents are processed to extract clean, normalized text
2. **Chunking**: Text is divided into semantic chunks with context preservation
3. **Embedding**: Chunks are converted to vector embeddings using SentenceTransformer
4. **Storage**: Embeddings and metadata are stored in Pinecone
5. **Retrieval**: User queries are converted to embeddings and matched against stored vectors
6. **Answer Generation**: Retrieved context is sent to LLM to generate relevant answers
