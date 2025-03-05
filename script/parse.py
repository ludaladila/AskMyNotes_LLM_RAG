import os
import glob
from typing import List, Dict
import fitz  
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 

class PDFTextExtractor:
  
    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "extracted_text")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_pdf_files(self) -> List[str]:
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))
        pdf_files.extend(glob.glob(os.path.join(self.input_dir, "*.PDF")))
        
        print(f"Found {len(pdf_files)} PDF files in directory {self.input_dir}")
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        filename = os.path.basename(pdf_path)
        result = {
            "filename": filename,
            "path": pdf_path,
            "success": False,
            "text": "",
            "pages": 0,
            "error": None
        }
        
        try:
            doc = fitz.open(pdf_path)
            result["pages"] = len(doc)
            
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Use "text" mode to extract plain text, ignoring tables and images
                page_text = page.get_text("text")
                full_text += page_text + "\n\n"  # Add line breaks to separate pages
            
            # Clean the text
            full_text = self.clean_text(full_text)
            
            result["text"] = full_text
            result["success"] = True
            
            # Close the document
            doc.close()
            
        except Exception as e:
            error_msg = f"Error extracting {filename}: {str(e)}"
            print(error_msg)
            result["error"] = error_msg
        
        return result
    
    def clean_text(self, text: str) -> str:
        # Remove consecutive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove unprintable characters, but keep Chinese, English, numbers and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:()\'"，。！？、；：《》【】「」\s]', '', text)
        
        # Merge multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing issues between Chinese and English
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def save_extracted_text(self, extraction_result: Dict) -> None:
        """Save the extracted text to a file"""
        if not extraction_result["success"]:
            return
        
        # Create output filename based on original filename
        base_name = os.path.splitext(extraction_result["filename"])[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.txt")
        
        # Write to text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extraction_result["text"])
        
        print(f"Saved extracted text to {output_path}")
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file and save results"""
        extraction_result = self.extract_text_from_pdf(pdf_path)
        
        if extraction_result["success"]:
            self.save_extracted_text(extraction_result)
            print(f"Successfully processed {extraction_result['filename']} ({extraction_result['pages']} pages)")
        else:
            print(f"Failed to process {extraction_result['filename']}: {extraction_result['error']}")
        
        return extraction_result
    
    def extract_all_pdfs(self, max_workers: int = 4) -> List[Dict]:
        pdf_files = self.get_pdf_files()
        results = []
        
        if not pdf_files:
            print("No PDF files found")
            return results
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to create a progress bar
            for result in tqdm(executor.map(self.process_single_pdf, pdf_files), 
                              total=len(pdf_files), 
                              desc="Processing PDF files"):
                results.append(result)
        
        # Count successful and failed processes
        success_count = sum(1 for r in results if r["success"])
        fail_count = len(results) - success_count
        
        print(f"PDF processing completed: {success_count} successful, {fail_count} failed")
        
        return results

# Usage example
if __name__ == "__main__":
    # Configure input and output directories
    INPUT_DIR = "../data"
    OUTPUT_DIR = "../data"
    
    # Create extractor instance
    extractor = PDFTextExtractor(INPUT_DIR, OUTPUT_DIR)
    
    # Execute extraction
    results = extractor.extract_all_pdfs(max_workers=4)  # Use 4 threads for parallel processing
    
    # Print summary
    print(f"\nProcessed {len(results)} PDF files in total")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")