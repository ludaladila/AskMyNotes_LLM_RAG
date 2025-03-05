import os
import re
from typing import List, Dict
from tqdm import tqdm

class SimpleTextChunker:
    def __init__(self, 
                 chunk_size: int = 200,
                 chunk_overlap: int = 20,
                 recursive: bool = False,
                 max_recursion_depth: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive = recursive
        self.max_recursion_depth = max_recursion_depth
    
    def is_mainly_chinese(self, text: str) -> bool:
        """Check if text is primarily Chinese"""
        if not text:
            return False
        
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_chars / len(text) > 0.5
    
    def simple_chunk_with_overlap(self, text: str, source: str) -> List[Dict]:
        chunks = []
        
        # Check if we should try to split on paragraph boundaries
        paragraphs = []
        if '\n\n' in text:
            # Split by double newlines to get paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If we have meaningful paragraphs, use them as base units
        if paragraphs and len(paragraphs) > 1 and max(len(p) for p in paragraphs) < self.chunk_size:
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                # If adding this paragraph would exceed the chunk size and we already have content
                if current_size + para_size > self.chunk_size and current_chunk:
                    # Create a chunk from what we have so far
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        "source": source,
                        "content": chunk_text,
                        "chunk_index": len(chunks),
                        "is_chinese": self.is_mainly_chinese(chunk_text)
                    })
                    
                    # Calculate how many paragraphs to keep for overlap
                    overlap_size = 0
                    overlap_paras = []
                    
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) <= self.chunk_overlap:
                            overlap_paras.insert(0, p)
                            overlap_size += len(p)
                        else:
                            break
                    
                    # Start the next chunk with the overlap paragraphs
                    current_chunk = overlap_paras
                    current_size = overlap_size
                
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_size
            
            # Add the last chunk if there's anything left
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "source": source,
                    "content": chunk_text,
                    "chunk_index": len(chunks),
                    "is_chinese": self.is_mainly_chinese(chunk_text)
                })
        else:
            # Fall back to character-based chunking
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_start = i
                chunk_end = min(i + self.chunk_size, len(text))
                
                if chunk_end <= chunk_start:
                    break
                    
                chunk_text = text[chunk_start:chunk_end]
                
                chunks.append({
                    "source": source,
                    "content": chunk_text,
                    "chunk_index": len(chunks),
                    "is_chinese": self.is_mainly_chinese(chunk_text)
                })
        
        return chunks
    
    def recursive_chunk(self, text: str, source: str, depth: int = 0) -> List[Dict]:
        if len(text) <= self.chunk_size or depth >= self.max_recursion_depth:
            return [{
                "source": source,
                "content": text,
                "chunk_index": 0,
                "recursion_depth": depth,
                "is_chinese": self.is_mainly_chinese(text)
            }]
        
        # First level
        if depth == 0 and '\n#' in text:  # Markdown header format
            sections = re.split(r'\n(#+ )', text)
            if len(sections) > 1:
                # Recombine the headers with their content
                combined_sections = []
                for i in range(1, len(sections), 2):
                    if i+1 < len(sections):
                        combined_sections.append(sections[i] + sections[i+1])
                    else:
                        combined_sections.append(sections[i])
                
                # Recursively process each section
                all_chunks = []
                for i, section in enumerate(combined_sections):
                    section_chunks = self.recursive_chunk(section, source, depth + 1)
                    
                    # Update chunk indices
                    for j, chunk in enumerate(section_chunks):
                        chunk["chunk_index"] = len(all_chunks) + j
                        chunk["section_index"] = i
                    
                    all_chunks.extend(section_chunks)
                
                return all_chunks
        
        # If no natural sections or not at top level, use overlap chunking
        return self.simple_chunk_with_overlap(text, source)
    
    def process_document(self, document: Dict) -> List[Dict]:
        if not document.get("text") or not document.get("success", False):
            print(f"Skipping document {document.get('filename', 'unknown')}: No text or extraction failed")
            return []
        
        text = document["text"]
        source = document.get("filename", "unknown")
        
        if self.recursive:
            chunks = self.recursive_chunk(text, source)
        else:
            chunks = self.simple_chunk_with_overlap(text, source)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk["document_pages"] = document.get("pages", 0)
            chunk["total_chunks"] = len(chunks)
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
        
        for doc in tqdm(documents, desc="Chunking documents"):
            doc_chunks = self.process_document(doc)
            all_chunks.extend(doc_chunks)
            
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i+1}/{len(chunks)}\n")
                f.write(f"Source: {chunk['source']}\n")
                f.write(f"Index: {chunk['chunk_index']}/{chunk['total_chunks']}\n")
                if "recursion_depth" in chunk:
                    f.write(f"Depth: {chunk['recursion_depth']}\n")
                f.write(f"Chinese: {chunk.get('is_chinese', False)}\n")
                f.write("Content:\n")
                f.write(chunk['content'])
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"Saved {len(chunks)} chunks to {output_path}")