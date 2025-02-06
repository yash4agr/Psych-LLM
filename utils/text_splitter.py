import json
from typing import Dict, List, Tuple
import numpy as np
import re
from config import VectorDBConfig

class TextChunker:
    def __init__(self, config: VectorDBConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
    def create_chunks_with_metadata(self, text: str, sections_metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Split text into overlapping chunks while preserving section context and page numbers.
        
        Args:
            text: String containing the full text with page markers
            sections_metadata: Dictionary containing section and page information
        
        Returns:
            List of tuples containing (chunk_text, metadata)
        """
        # Split text into pages first
        pages = self._split_into_pages(text)
        chunks = []
        
        current_chunk = []
        current_chunk_size = 0
        current_page = 1
        
        for page_num, page_text in pages.items():
            # Clean the text
            cleaned_text = self._clean_text(page_text)
            words = cleaned_text.split()
            
            i = 0
            while i < len(words):
                # If current chunk is empty, start new chunk
                if not current_chunk:
                    metadata = self.get_section_for_page(page_num, sections_metadata)
                    current_page = page_num
                
                # Add words to current chunk until reaching chunk_size
                while i < len(words) and current_chunk_size < self.chunk_size:
                    current_chunk.append(words[i])
                    current_chunk_size += 1
                    i += 1
                
                # If chunk is full or we're at end of page, save it
                if current_chunk_size >= self.chunk_size or i >= len(words):
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((
                        chunk_text,
                        {
                            'section': metadata['section'],
                            'subsection': metadata['subsection'],
                            'page': current_page
                        }
                    ))
                    
                    # Start new chunk with overlap
                    if i < len(words):
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                        current_chunk_size = len(current_chunk)
                    else:
                        current_chunk = []
                        current_chunk_size = 0
            
        return chunks

    def _split_into_pages(self, text: str) -> Dict[int, str]:
        """Split text into pages based on page markers."""
        pages = {}
        current_page = None
        current_text = []
        
        for line in text.split('\n'):
            # Check for page marker (assuming format like [PAGE 1] or similar)
            page_match = re.match(r'\[PAGE (\d+)\]', line)
            if page_match:
                if current_page is not None:
                    pages[current_page] = '\n'.join(current_text)
                current_page = int(page_match.group(1))
                current_text = []
            else:
                current_text.append(line)
        
        # Don't forget to add the last page
        if current_page is not None:
            pages[current_page] = '\n'.join(current_text)
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        return text

    def get_section_for_page(self, page_num: int, sections_metadata: Dict) -> Dict:
        for section, data in sections_metadata.items():
            if data['page_start'] <= page_num <= data['page_end']:
                for subsection, sub_data in data.get('subsections', {}).items():
                    if sub_data['page_start'] <= page_num <= sub_data['page_end']:
                        return {
                            'section': section,
                            'subsection': subsection,
                            'page': page_num
                        }
        return {'section': 'Unknown', 'subsection': 'Unknown', 'page': page_num}