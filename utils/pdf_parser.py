import pdfplumber

class PDFParser:
    def __init__(self):
        self.current_page = 0
    
    def parse_pdf(self, pdf_path: str) -> str:
        """
        Parse PDF and return text with page markers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            String containing all text with page markers
        """
        extracted_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text from page
                text = page.extract_text()
                if text:
                    # Add page marker
                    extracted_text.append(f"[PAGE {page_num}]")
                    extracted_text.append(text.strip())
        
        return '\n'.join(extracted_text)