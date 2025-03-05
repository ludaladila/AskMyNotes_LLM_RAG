from embedding import get_embedding
from vector import VectorStore
from chunk import SimpleTextChunker
from parse import PDFTextExtractor

def build_knowledge_base(pdf_folder):
    extractor = PDFTextExtractor(pdf_folder)
    documents = extractor.extract_all_pdfs()

    chunker = SimpleTextChunker()
    all_chunks = chunker.process_documents(documents)

    store = VectorStore()
    embeddings = [get_embedding(chunk["content"]) for chunk in all_chunks]

    store.add(embeddings, all_chunks)

    print(f"✅ Knowledge base built with {len(all_chunks)} chunks.")
    return store
