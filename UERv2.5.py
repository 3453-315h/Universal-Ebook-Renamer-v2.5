#!/usr/bin/env python3
"""
Universal Ebook Renamer - FIX: Added online data preference option
Supports PDF, EPUB, MOBI, FB2, AZW, AZW3 with Google Books API integration
"""

import os
import re
import shutil
import argparse
import logging
import zipfile
import xml.etree.ElementTree as ET
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set
from datetime import datetime

# 🔧 FIXED: Auto-install requests if missing
try:
    import requests
except ImportError:
    print("📦 Installing requests...")
    os.system("pip install requests")
    import requests

try:
    import PyPDF2
except ImportError:
    print("📦 Installing PyPDF2...")
    os.system("pip install PyPDF2")
    import PyPDF2

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def disable():
        Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BLUE = Colors.CYAN = Colors.MAGENTA = Colors.BOLD = Colors.RESET = ''
    
    @staticmethod
    def enable():
        Colors.RED = '\033[91m'
        Colors.GREEN = '\033[92m'
        Colors.YELLOW = '\033[93m'
        Colors.BLUE = '\033[94m'
        Colors.CYAN = '\033[96m'
        Colors.MAGENTA = '\033[95m'
        Colors.BOLD = '\033[1m'
        Colors.RESET = '\033[0m'


class EbookMetadataExtractor:
    """Base class for ebook metadata extraction."""
    
    def __init__(self, template: str = "{title}", case_style: str = "original", 
                 max_len: int = 100, space_replacement: str = "none", use_ocr: bool = True,
                 use_online_search: bool = False, online_prefer: bool = False, verbose: bool = True,
                 replace_and_with_ampersand: bool = False):  # NEW PARAMETER
        self.template = template
        self.case_style = case_style
        self.max_len = max_len
        self.space_replacement = space_replacement
        self.use_ocr = use_ocr
        self.use_online_search = use_online_search
        self.online_prefer = online_prefer
        self.verbose = verbose
        self.replace_and_with_ampersand = replace_and_with_ampersand  # NEW
        self.title_cache = {}
        self.online_cache = {}
    
    def _log(self, message: str, color: str = "") -> None:
        """Unified logging method that respects verbose setting."""
        if self.verbose:
            reset = Colors.RESET if hasattr(Colors, 'RESET') and Colors.RESET else ''
            print(f"{color}{message}{reset}")
    
    def _clean_text(self, text: any) -> Optional[str]:
        """Clean and validate extracted text."""
        if not isinstance(text, str):
            return None
        
        cleaned = text.strip()
        return cleaned if len(cleaned) > 2 else None
    
    def _clean_year(self, year_str: str) -> str:
        """Clean year string by replacing common OCR errors and extracting digits."""
        if not year_str:
            return ""
        replacements = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        for old, new in replacements.items():
            year_str = year_str.replace(old, new)
        digits = ''.join(c for c in year_str if c.isdigit())
        return digits[:4] if len(digits) >= 4 else year_str
    
    def _extract_year_from_str(self, text: any) -> Optional[str]:
        """Extract year from string - base implementation"""
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return year_match.group()
        return None
    
    def _validate_isbn(self, isbn: str) -> bool:
        """Validate ISBN - base implementation"""
        if len(isbn) not in [10, 13]:
            return False
        
        if len(isbn) == 13 and isbn.isdigit():
            return isbn.startswith(('978', '979'))
        
        if len(isbn) == 10:
            return isbn[:-1].isdigit() and (isbn[-1].isdigit() or isbn[-1].upper() == 'X')
        
        return False
    
    def _cross_reference_google_books(self, metadata: Dict[str, Optional[str]], file_path: Path) -> Dict[str, Optional[str]]:
        """Search Google Books API to enhance metadata."""
        if not self.use_online_search:
            return metadata
        
        cache_key = metadata.get('isbn') or metadata.get('title', '')
        if not cache_key:
            return metadata
        
        if cache_key in self.online_cache:
            self._log(f"  🔍 Using cached online data", Colors.CYAN)
            online_meta = self.online_cache[cache_key]
            
            if self.online_prefer and online_meta and online_meta.get('title'):
                self._log(f"  🌐 Using online data exclusively", Colors.MAGENTA)
                online_meta_copy = online_meta.copy()
                online_meta_copy['_attempts'] = metadata.get('_attempts', [])
                online_meta_copy['_sources'] = {**metadata.get('_sources', {}), **online_meta_copy.get('_sources', {})}
                return online_meta_copy
            
            return self._merge_metadata(metadata, online_meta, 'google_books_cache')
        
        try:
            time.sleep(1)
            
            if metadata.get('isbn'):
                query = f'isbn:{metadata["isbn"]}'
            elif metadata.get('title') and metadata.get('author'):
                query = f'intitle:{metadata["title"]}+inauthor:{metadata["author"]}'
            elif metadata.get('title'):
                query = f'intitle:{metadata["title"]}'
            else:
                return metadata
            
            response = requests.get(
                'https://www.googleapis.com/books/v1/volumes',
                params={'q': query, 'maxResults': 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'items' in data and len(data['items']) > 0:
                    book_info = data['items'][0]['volumeInfo']
                    
                    online_meta = {
                        'title': book_info.get('title'),
                        'author': ', '.join(book_info.get('authors', [])),
                        'year': book_info.get('publishedDate', '')[:4] if book_info.get('publishedDate') else None,
                        'isbn': None,
                        '_sources': {'_source': 'google_books'}
                    }
                    
                    for identifier in book_info.get('industryIdentifiers', []):
                        if identifier.get('type') in ['ISBN_13', 'ISBN_10']:
                            online_meta['isbn'] = identifier.get('identifier')
                            break
                    
                    self.online_cache[cache_key] = online_meta
                    
                    self._log(f"  🔍 Found online: {online_meta.get('title', '')[:50]}...", Colors.CYAN)
                    
                    if self.online_prefer:
                        self._log(f"  🌐 Using online data exclusively", Colors.MAGENTA)
                        online_meta_copy = online_meta.copy()
                        online_meta_copy['_attempts'] = metadata.get('_attempts', [])
                        return online_meta_copy
                    
                    return self._merge_metadata(metadata, online_meta, 'google_books')
            
        except Exception as e:
            self._log(f"  ⚠️ Online search failed: {e}", Colors.YELLOW)
        
        return metadata
    
    def _merge_metadata(self, local_meta: Dict, online_meta: Dict, source: str) -> Dict:
        """Merge local and online metadata based on preference."""
        if self.online_prefer and online_meta and online_meta.get('title'):
            merged = local_meta.copy()
            for key in ['title', 'author', 'year', 'isbn']:
                if online_meta.get(key):
                    merged[key] = online_meta[key]
                    merged['_sources'][key] = source
            return merged
        
        merged = local_meta.copy()
        for key in ['title', 'author', 'year', 'isbn']:
            if not merged.get(key) and online_meta.get(key):
                merged[key] = online_meta[key]
                merged['_sources'][key] = source
        
        return merged
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Optional[str]]:
        raise NotImplementedError
    
    def format_filename(self, metadata: Dict[str, Optional[str]]) -> str:
        """Generate filename from metadata using template."""
        safe_meta = {}
        for key, value in metadata.items():
            if not key.startswith('_') and value:
                if self.case_style == "lower":
                    safe_meta[key] = value.lower()
                elif self.case_style == "upper":
                    safe_meta[key] = value.upper()
                elif self.case_style == "title":
                    safe_meta[key] = self._to_title_case(value)
                else:
                    safe_meta[key] = value
                
                # NEW: Replace "and" with "&" in title if enabled
                if key == 'title' and self.replace_and_with_ampersand:
                    safe_meta[key] = re.sub(r'\band\b', '&', safe_meta[key], flags=re.IGNORECASE)
                
                safe_meta[key] = self._sanitize_part(safe_meta[key])
            elif not key.startswith('_'):
                safe_meta[key] = ""
        
        try:
            class SafeFormatDict(dict):
                def __missing__(self, key):
                    return ""
            
            format_dict = SafeFormatDict(safe_meta)
            filename = self.template.format_map(format_dict)
        except Exception as e:
            self._log(f"  ⚠️ Template error: {e}")
            filename = safe_meta.get('title', 'untitled')
        
        return self._finalize_filename(filename)
    
    def _sanitize_part(self, text: str) -> str:
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        return ' '.join(text.split())
    
    def _to_title_case(self, text: str) -> str:
        if not text:
            return ""
        words = text.split()
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of', 'with'}
        result = []
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word.lower() not in small_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        return ' '.join(result)
    
    def _finalize_filename(self, filename: str) -> str:
        filename = re.sub(r'\s+', ' ', filename)
        filename = filename.strip(' -_.')
        
        if not filename:
            filename = "untitled_document"
        
        if self.space_replacement == "underscore":
            filename = filename.replace(" ", "_")
        elif self.space_replacement == "dash":
            filename = filename.replace(" ", "-")
        elif self.space_replacement == "dot":
            filename = filename.replace(" ", ".")
        
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        if len(filename) > self.max_len:
            if ' - ' in filename:
                parts = filename.split(' - ', 1)
                available = self.max_len - len(parts[1]) - 3
                filename = parts[0][:available] + '...' + parts[1] if available > 10 else filename[:self.max_len-3] + "..."
            else:
                filename = filename[:self.max_len-3] + "..."
        
        return filename.strip()


class PdfMetadataExtractor(EbookMetadataExtractor):
    """PDF metadata extractor using PyPDF2."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.year_patterns = [
            re.compile(r'\b(19|20)\d{2}\b'),
            re.compile(r'\b(19|20)[\dOIl]{2}\b'),
            re.compile(r'(?:©|copyright|\(c\))\s*[\dOIl]{4}', re.IGNORECASE),
            re.compile(r'(?:published|issued|release[d]?|printed)\s*[\dOIl]{4}', re.IGNORECASE),
            re.compile(r'\([\dOIl]{4}\)'),
        ]
        
        self.isbn_patterns = [
            re.compile(r'ISBN-13[\s:-]*([0-9]{1,5}[\s-][0-9]{1,7}[\s-][0-9]{1,7}[\s-][0-9]{1,3})', re.IGNORECASE),
            re.compile(r'ISBN-10[\s:-]*([0-9]{1,5}[\s-][0-9]{1,7}[\s-][0-9]{1,7}[\s-][0-9Xx]{1,3})', re.IGNORECASE),
            re.compile(r'ISBN[\s:-]*([0-9-Xx\s]{10,17})', re.IGNORECASE),
            re.compile(r'\b(97[89][\d\s-]{11,15})\b'),
            re.compile(r'\b(97[89][\dOIl]{11})\b'),
            re.compile(r'\b([0-9]{9}[0-9Xx])\b'),
        ]
        
        self.author_patterns = [
            re.compile(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
            re.compile(r'\bwritten\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
            re.compile(r'\bauthor[s]?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
            re.compile(r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$', re.MULTILINE),
        ]
    
    def _fix_ocr_errors(self, text: str) -> str:
        if not self.use_ocr:
            return text
        
        replacements = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            if any(keyword in line.upper() for keyword in ['ISBN', '978', '979', 'COPYRIGHT', '©']):
                for old, new in replacements.items():
                    line = re.sub(rf'(?<=\d){old}(?=\d)', new, line)
                    line = re.sub(rf'(?<=\d){old}$', new, line)
                    line = re.sub(rf'^{old}(?=\d)', new, line)
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _extract_year_from_str(self, text: any) -> Optional[str]:
        """Extract year from PDF metadata date strings"""
        if not isinstance(text, str):
            return None
        
        # Handle PDF date format: D:20211214120000+00'00'
        if text.startswith('D:'):
            text = text[2:]
        
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return year_match.group()
        return None
    
    def _extract_year_deep(self, pages_text: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        year_candidates = []
        
        for text, page_num in pages_text:
            context_score = self._score_year_context(text)
            for pattern in self.year_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    year_str = self._clean_year(match.group())
                    if len(year_str) >= 4 and year_str.isdigit():
                        year = int(year_str[:4])
                        current_year = datetime.now().year + 2
                        if 1900 <= year <= current_year:
                            page_score = max(0, 10 - page_num)
                            total_score = context_score + page_score
                            year_candidates.append((year, page_num, total_score))
        
        if year_candidates:
            year_candidates.sort(key=lambda x: x[2], reverse=True)
            best_year, page_num, _ = year_candidates[0]
            return str(best_year), page_num
        
        return None
    
    def _score_year_context(self, text: str) -> int:
        score = 0
        lower_text = text.lower()
        if any(marker in lower_text for marker in ['©', 'copyright', '(c)', 'published', 'printed']):
            score += 10
        if any(marker in lower_text for marker in ['publication date', 'release date', 'edition', 'version']):
            score += 5
        if any(marker in lower_text for marker in ['journal', 'conference', 'proceedings']):
            score += 3
        return score
    
    def _extract_isbn_deep(self, pages_text: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        isbn_candidates = []
        
        for text, page_num in pages_text:
            for pattern in self.isbn_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    groups = match.groups()
                    isbn_str = groups[0] if groups else match.group()
                    isbn_clean = self._clean_isbn(isbn_str)
                    
                    if self._validate_isbn(isbn_clean):
                        score = 10 - page_num
                        if len(isbn_clean) == 13:
                            score += 5
                        isbn_candidates.append((isbn_clean, page_num, score))
        
        if isbn_candidates:
            isbn_candidates.sort(key=lambda x: x[2], reverse=True)
            best_isbn, page_num, _ = isbn_candidates[0]
            return best_isbn, page_num
        
        return None
    
    def _clean_isbn(self, isbn_str: str) -> str:
        isbn_clean = re.sub(r'[-\s]', '', isbn_str)
        replacements = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        for old, new in replacements.items():
            isbn_clean = isbn_clean.replace(old, new)
        return isbn_clean
    
    def _extract_title_deep(self, pages_text: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        strategies = [
            self._title_strategy_title_case,
            self._title_strategy_all_caps,
            self._title_strategy_first_line,
            self._title_strategy_longest_line
        ]
        
        for strategy in strategies:
            for text, page_num in pages_text:
                if page_num <= 5:
                    result = strategy(text)
                    if result and len(result) > 10:
                        return result, page_num
        
        return None
    
    def _title_strategy_title_case(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:
            if self._is_title_case(line) and len(line) > 15:
                return line
        return None
    
    def _title_strategy_all_caps(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:
            if line.isupper() and len(line) > 10 and not line.isdigit():
                return line
        return None
    
    def _title_strategy_first_line(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            if len(line) > 20 and not line.isdigit():
                return line
        return None
    
    def _title_strategy_longest_line(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()][:5]
        if lines:
            return max(lines, key=lambda x: len(x) if not x.isdigit() else 0)
        return None
    
    def _extract_author_deep(self, pages_text: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        for text, page_num in pages_text:
            if page_num <= 5:
                for pattern in self.author_patterns:
                    match = pattern.search(text)
                    if match:
                        author = match.group(1).strip()
                        if self._is_valid_author_name(author):
                            return author, page_num
        
        return None
    
    def _is_valid_author_name(self, name: str) -> bool:
        words = name.split()
        if len(words) < 2 or len(words) > 6:
            return False
        
        capital_count = sum(1 for w in words if w and w[0].isupper())
        if capital_count < len(words) - 1:
            return False
        
        false_positives = ['TABLE OF CONTENTS', 'PREFACE', 'INTRODUCTION', 'CHAPTER', 'APPENDIX']
        if name.upper() in false_positives:
            return False
        
        return any(self._looks_like_name(w) for w in words)
    
    def _looks_like_name(self, word: str) -> bool:
        if len(word) < 2:
            return False
        if not word[0].isupper():
            return False
        if len(word) == 2 and word[1] == '.':
            return False
        return True
    
    def _is_title_case(self, text: str) -> bool:
        words = text.split()
        if len(words) < 2:
            return False
        
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of', 'with'}
        capitalized = sum(1 for word in words if word and word[0].isupper() and word.lower() not in small_words)
        return capitalized / max(1, len(words) - words.count('')) > 0.5
    
    def _scan_pages(self, reader: PyPDF2.PdfReader, max_pages: int = 15) -> List[Tuple[str, int]]:
        pages_text = []
        for i, page in enumerate(reader.pages[:max_pages]):
            try:
                text = page.extract_text()
                if text and text.strip():
                    text = self._fix_ocr_errors(text)
                    pages_text.append((text, i + 1))
            except Exception:
                continue
        return pages_text
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Optional[str]]:
        if file_path in self.title_cache:
            return self.title_cache[file_path]
        
        metadata = {
            'title': None,
            'author': None,
            'year': None,
            'isbn': None,
            '_sources': {},
            '_attempts': []
        }
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                try:
                    pdf_meta = reader.metadata
                    if pdf_meta:
                        metadata['title'] = self._clean_text(pdf_meta.get('/Title', ''))
                        metadata['author'] = self._clean_text(pdf_meta.get('/Author', ''))
                        
                        dates = [pdf_meta.get('/CreationDate', ''), pdf_meta.get('/ModDate', '')]
                        for date_str in dates:
                            year = self._extract_year_from_str(date_str)
                            if year:
                                metadata['year'] = year
                                metadata['_sources']['year'] = 'metadata'
                                break
                except Exception as e:
                    self._log(f"  ⚠️ Metadata extraction failed: {e}, falling back to OCR...", Colors.YELLOW)
                    metadata['title'] = None
                    metadata['author'] = None
                    metadata['year'] = None
                
                pages_text = self._scan_pages(reader, max_pages=15)
                
                if not metadata['title']:
                    result = self._extract_title_deep(pages_text)
                    if result:
                        metadata['title'], page_num = result
                        metadata['_sources']['title'] = f'page_{page_num}'
                
                if not metadata['author']:
                    result = self._extract_author_deep(pages_text)
                    if result:
                        metadata['author'], page_num = result
                        metadata['_sources']['author'] = f'page_{page_num}'
                
                if not metadata['year']:
                    result = self._extract_year_deep(pages_text)
                    if result:
                        metadata['year'], page_num = result
                        metadata['_sources']['year'] = f'page_{page_num}'
                        metadata['_attempts'].append(f"Year found on page {page_num}")
                
                if not metadata['isbn']:
                    result = self._extract_isbn_deep(pages_text)
                    if result:
                        metadata['isbn'], page_num = result
                        metadata['_sources']['isbn'] = f'page_{page_num}'
                        metadata['_attempts'].append(f"ISBN found on page {page_num}")
                
                if self.use_online_search:
                    metadata = self._cross_reference_google_books(metadata, file_path)
                
        except Exception as e:
            self._log(f"  ⚠️ Error reading PDF: {e}", Colors.RED)
        
        self.title_cache[file_path] = metadata
        return metadata


class EpubMetadataExtractor(EbookMetadataExtractor):
    """EPUB metadata extractor using OPF XML."""
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Optional[str]]:
        if file_path in self.title_cache:
            return self.title_cache[file_path]
        
        metadata = {
            'title': None,
            'author': None,
            'year': None,
            'isbn': None,
            '_sources': {},
            '_attempts': []
        }
        
        try:
            with zipfile.ZipFile(file_path, 'r') as epub:
                opf_files = [f for f in epub.namelist() if f.endswith('.opf')]
                if not opf_files:
                    metadata['title'] = self._fallback_title(file_path)
                    return metadata
                
                opf_content = epub.read(opf_files[0]).decode('utf-8')
                root = ET.fromstring(opf_content)
                
                ns = {
                    'opf': 'http://www.idpf.org/2007/opf',
                    'dc': 'http://purl.org/dc/elements/1.1/'
                }
                
                title_elem = root.find('.//dc:title', ns)
                if title_elem is not None and title_elem.text:
                    metadata['title'] = title_elem.text.strip()
                    metadata['_sources']['title'] = 'opf_metadata'
                
                creator_elem = root.find('.//dc:creator', ns)
                if creator_elem is not None and creator_elem.text:
                    metadata['author'] = creator_elem.text.strip()
                    metadata['_sources']['author'] = 'opf_metadata'
                
                date_elem = root.find('.//dc:date', ns)
                if date_elem is not None and date_elem.text:
                    year = self._extract_year_from_str(date_elem.text)
                    if year:
                        metadata['year'] = year
                        metadata['_sources']['year'] = 'opf_metadata'
                
                identifier_elem = root.find('.//dc:identifier[@opf:scheme="ISBN"]', ns)
                if identifier_elem is not None and identifier_elem.text:
                    isbn_clean = re.sub(r'[-\s]', '', identifier_elem.text)
                    if self._validate_isbn(isbn_clean):
                        metadata['isbn'] = isbn_clean
                        metadata['_sources']['isbn'] = 'opf_metadata'
                
                if not metadata['title']:
                    metadata['title'] = self._fallback_title(file_path)
                
                if self.use_online_search:
                    metadata = self._cross_reference_google_books(metadata, file_path)
            
        except Exception as e:
            self._log(f"  ⚠️ Error reading EPUB: {e}", Colors.RED)
            metadata['title'] = self._fallback_title(file_path)
        
        self.title_cache[file_path] = metadata
        return metadata
    
    def _fallback_title(self, file_path: Path) -> str:
        title = file_path.stem
        title = re.sub(r'[\._-]', ' ', title)
        title = re.sub(r'\b(epub|mobi|azw|fb2|pdf)\b', '', title, flags=re.IGNORECASE)
        return title.strip()


class MobiMetadataExtractor(EbookMetadataExtractor):
    """MOBI/AZW/AZW3 metadata extractor with deep content scanning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.year_patterns = [
            re.compile(r'\b(19|20)\d{2}\b'),
            re.compile(r'(?:©|copyright|\(c\))\s*[\dOIl]{4}', re.IGNORECASE),
            re.compile(r'(?:published|issued|release[d]?|printed)\s*[\dOIl]{4}', re.IGNORECASE),
        ]
        
        self.isbn_patterns = [
            re.compile(r'ISBN-13[\s:-]*([0-9]{1,5}[\s-][0-9]{1,7}[\s-][0-9]{1,7}[\s-][0-9]{1,3})', re.IGNORECASE),
            re.compile(r'ISBN-10[\s:-]*([0-9]{1,5}[\s-][0-9]{1,7}[\s-][0-9]{1,7}[\s-][0-9Xx]{1,3})', re.IGNORECASE),
            re.compile(r'ISBN[\s:-]*([0-9-Xx\s]{10,17})', re.IGNORECASE),
            re.compile(r'\b(97[89][\d\s-]{11,15})\b'),
            re.compile(r'\b([0-9]{9}[0-9Xx])\b'),
        ]
        
        self.author_patterns = [
            re.compile(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
            re.compile(r'\bwritten\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
            re.compile(r'\bauthor[s]?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', re.IGNORECASE),
        ]
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract readable text from MOBI/AZW file."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            text_candidates = []
            
            # Try header metadata first
            if b'TITLE' in content[:1024]:
                try:
                    title_match = re.search(b'TITLE\x00([^\x00]+)', content[:1024])
                    if title_match:
                        text_candidates.append(f"HEADER_TITLE: {title_match.group(1).decode('utf-8', errors='ignore').strip()}")
                except:
                    pass
            
            if b'AUTHOR' in content[:1024]:
                try:
                    author_match = re.search(b'AUTHOR\x00([^\x00]+)', content[:1024])
                    if author_match:
                        text_candidates.append(f"HEADER_AUTHOR: {author_match.group(1).decode('utf-8', errors='ignore').strip()}")
                except:
                    pass
            
            # Extract all readable ASCII text (32-126) sequences
            text_chunks = re.findall(b'[\x20-\x7e]{10,}', content)
            for chunk in text_chunks[:100]:
                try:
                    decoded = chunk.decode('utf-8', errors='ignore')
                    if len(decoded) > 20:
                        text_candidates.append(decoded)
                except:
                    continue
            
            return '\n'.join(text_candidates)
            
        except Exception as e:
            self._log(f"  ⚠️ Text extraction error: {e}", Colors.YELLOW)
            return ""
    
    def _extract_title_deep(self, lines: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        strategies = [
            self._title_strategy_title_case,
            self._title_strategy_all_caps,
            self._title_strategy_first_line,
            self._title_strategy_longest_line
        ]
        
        for strategy in strategies:
            for text, page_num in lines:
                result = strategy(text)
                if result and len(result) > 10:
                    return result, page_num
        
        return None
    
    def _title_strategy_title_case(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:
            if self._is_title_case(line) and len(line) > 15:
                return line
        return None
    
    def _title_strategy_all_caps(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:
            if line.isupper() and len(line) > 10 and not line.isdigit():
                return line
        return None
    
    def _title_strategy_first_line(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            if len(line) > 20 and not line.isdigit():
                return line
        return None
    
    def _title_strategy_longest_line(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()][:5]
        if lines:
            return max(lines, key=lambda x: len(x) if not x.isdigit() else 0)
        return None
    
    def _extract_author_deep(self, lines: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        for text, page_num in lines:
            for pattern in self.author_patterns:
                match = pattern.search(text)
                if match:
                    author = match.group(1).strip()
                    if self._is_valid_author_name(author):
                        return author, page_num
        return None
    
    def _is_valid_author_name(self, name: str) -> bool:
        words = name.split()
        if len(words) < 2 or len(words) > 6:
            return False
        
        capital_count = sum(1 for w in words if w and w[0].isupper())
        if capital_count < len(words) - 1:
            return False
        
        false_positives = ['TABLE OF CONTENTS', 'PREFACE', 'INTRODUCTION', 'CHAPTER', 'APPENDIX']
        if name.upper() in false_positives:
            return False
        
        return any(self._looks_like_name(w) for w in words)
    
    def _looks_like_name(self, word: str) -> bool:
        if len(word) < 2:
            return False
        if not word[0].isupper():
            return False
        if len(word) == 2 and word[1] == '.':
            return False
        return True
    
    def _extract_year_deep(self, lines: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        year_candidates = []
        
        for text, page_num in lines:
            context_score = self._score_year_context(text)
            for pattern in self.year_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    year_str = self._clean_year(match.group())
                    if len(year_str) >= 4 and year_str.isdigit():
                        year = int(year_str[:4])
                        current_year = datetime.now().year + 2
                        if 1900 <= year <= current_year:
                            total_score = context_score
                            year_candidates.append((year, page_num, total_score))
        
        if year_candidates:
            year_candidates.sort(key=lambda x: x[2], reverse=True)
            best_year, page_num, _ = year_candidates[0]
            return str(best_year), page_num
        
        return None
    
    def _score_year_context(self, text: str) -> int:
        score = 0
        lower_text = text.lower()
        if any(marker in lower_text for marker in ['©', 'copyright', '(c)', 'published', 'printed']):
            score += 10
        return score
    
    def _extract_isbn_deep(self, lines: List[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
        isbn_candidates = []
        
        for text, page_num in lines:
            for pattern in self.isbn_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    groups = match.groups()
                    isbn_str = groups[0] if groups else match.group()
                    isbn_clean = self._clean_isbn(isbn_str)
                    
                    if self._validate_isbn(isbn_clean):
                        score = 5
                        if len(isbn_clean) == 13:
                            score += 5
                        isbn_candidates.append((isbn_clean, page_num, score))
        
        if isbn_candidates:
            isbn_candidates.sort(key=lambda x: x[2], reverse=True)
            best_isbn, page_num, _ = isbn_candidates[0]
            return best_isbn, page_num
        
        return None
    
    def _clean_isbn(self, isbn_str: str) -> str:
        isbn_clean = re.sub(r'[-\s]', '', isbn_str)
        replacements = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        for old, new in replacements.items():
            isbn_clean = isbn_clean.replace(old, new)
        return isbn_clean
    
    def _validate_isbn(self, isbn: str) -> bool:
        if len(isbn) not in [10, 13]:
            return False
        
        if len(isbn) == 13 and isbn.isdigit():
            return isbn.startswith(('978', '979'))
        
        if len(isbn) == 10:
            return isbn[:-1].isdigit() and (isbn[-1].isdigit() or isbn[-1].upper() == 'X')
        
        return False
    
    def _is_title_case(self, text: str) -> bool:
        words = text.split()
        if len(words) < 2:
            return False
        
        small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of', 'with'}
        capitalized = sum(1 for word in words if word and word[0].isupper() and word.lower() not in small_words)
        return capitalized / max(1, len(words) - words.count('')) > 0.5
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Optional[str]]:
        if file_path in self.title_cache:
            return self.title_cache[file_path]
        
        metadata = {
            'title': None,
            'author': None,
            'year': None,
            'isbn': None,
            '_sources': {},
            '_attempts': []
        }
        
        try:
            full_text = self._extract_text_content(file_path)
            
            if not full_text:
                raise Exception("No readable text extracted")
            
            lines = [(line, 0) for line in full_text.split('\n') if line.strip()]
            
            # Try header first
            header_title = re.search(r'HEADER_TITLE:\s*(.+)', full_text)
            if header_title:
                metadata['title'] = self._clean_text(header_title.group(1))
                metadata['_sources']['title'] = 'mobi_header'
            
            header_author = re.search(r'HEADER_AUTHOR:\s*(.+)', full_text)
            if header_author:
                metadata['author'] = self._clean_text(header_author.group(1))
                metadata['_sources']['author'] = 'mobi_header'
            
            # Deep extraction
            if not metadata['title']:
                result = self._extract_title_deep(lines)
                if result:
                    metadata['title'], _ = result
                    metadata['_sources']['title'] = 'mobi_content_scan'
            
            if not metadata['author']:
                result = self._extract_author_deep(lines)
                if result:
                    metadata['author'], _ = result
                    metadata['_sources']['author'] = 'mobi_content_scan'
            
            if not metadata['year']:
                result = self._extract_year_deep(lines)
                if result:
                    metadata['year'], _ = result
                    metadata['_sources']['year'] = 'mobi_content_scan'
                    metadata['_attempts'].append(f"Year found in content scan")
            
            if not metadata['isbn']:
                result = self._extract_isbn_deep(lines)
                if result:
                    metadata['isbn'], _ = result
                    metadata['_sources']['isbn'] = 'mobi_content_scan'
                    metadata['_attempts'].append(f"ISBN found in content scan")
            
            if not metadata['title']:
                metadata['title'] = self._fallback_title(file_path)
                
            if self.use_online_search:
                metadata = self._cross_reference_google_books(metadata, file_path)
            
        except Exception as e:
            self._log(f"  ⚠️ Error reading MOBI/AZW: {e}", Colors.RED)
            metadata['title'] = self._fallback_title(file_path)
        
        self.title_cache[file_path] = metadata
        return metadata
    
    def _fallback_title(self, file_path: Path) -> str:
        title = file_path.stem
        title = re.sub(r'[\._-]', ' ', title)
        title = re.sub(r'\b(mobi|azw|azw3|epub|fb2|pdf)\b', '', title, flags=re.IGNORECASE)
        return title.strip()


class Fb2MetadataExtractor(EbookMetadataExtractor):
    """FB2 (FictionBook) metadata extractor using XML."""
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Optional[str]]:
        if file_path in self.title_cache:
            return self.title_cache[file_path]
        
        metadata = {
            'title': None,
            'author': None,
            'year': None,
            'isbn': None,
            '_sources': {},
            '_attempts': []
        }
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
            
            title_elem = root.find('.//fb:book-title', ns)
            if title_elem is not None and title_elem.text:
                metadata['title'] = title_elem.text.strip()
                metadata['_sources']['title'] = 'fb2_metadata'
            
            author_first = root.find('.//fb:first-name', ns)
            author_last = root.find('.//fb:last-name', ns)
            if author_first is not None and author_last is not None:
                author_parts = []
                if author_first.text:
                    author_parts.append(author_first.text.strip())
                if author_last.text:
                    author_parts.append(author_last.text.strip())
                if author_parts:
                    metadata['author'] = ' '.join(author_parts)
                    metadata['_sources']['author'] = 'fb2_metadata'
            
            date_elem = root.find('.//fb:date', ns)
            if date_elem is not None and date_elem.text:
                year = self._extract_year_from_str(date_elem.text)
                if year:
                    metadata['year'] = year
                    metadata['_sources']['year'] = 'fb2_metadata'
            
            isbn_elem = root.find('.//fb:isbn', ns)
            if isbn_elem is not None and isbn_elem.text:
                isbn_clean = re.sub(r'[-\s]', '', isbn_elem.text)
                if self._validate_isbn(isbn_clean):
                    metadata['isbn'] = isbn_clean
                    metadata['_sources']['isbn'] = 'fb2_metadata'
            
            if not metadata['title']:
                metadata['title'] = self._fallback_title(file_path)
            
            if self.use_online_search:
                metadata = self._cross_reference_google_books(metadata, file_path)
            
        except Exception as e:
            self._log(f"  ⚠️ Error reading FB2: {e}", Colors.RED)
            metadata['title'] = self._fallback_title(file_path)
        
        self.title_cache[file_path] = metadata
        return metadata
    
    def _fallback_title(self, file_path: Path) -> str:
        title = file_path.stem
        title = re.sub(r'[\._-]', ' ', title)
        title = re.sub(r'\b(fb2|epub|mobi|azw|pdf)\b', '', title, flags=re.IGNORECASE)
        return title.strip()


def get_extractor(file_path: Path, verbose: bool = True, **kwargs) -> EbookMetadataExtractor:
    """Factory function to get appropriate extractor."""
    ext = file_path.suffix.lower()
    
    if ext == '.pdf':
        return PdfMetadataExtractor(verbose=verbose, **kwargs)
    elif ext == '.epub':
        return EpubMetadataExtractor(verbose=verbose, **kwargs)
    elif ext in ['.mobi', '.azw', '.azw3']:
        return MobiMetadataExtractor(verbose=verbose, **kwargs)
    elif ext == '.fb2':
        return Fb2MetadataExtractor(verbose=verbose, **kwargs)
    else:
        raise ValueError(f"❌ Unsupported format: {ext}")


def create_backup(file_path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / file_path.name
    if not backup_path.exists():
        shutil.copy2(file_path, backup_dir)


def get_formats_to_process(args) -> Set[str]:
    """Determine which file formats to process based on CLI arguments."""
    # If any format is explicitly enabled, only process those
    enabled_formats = {
        'pdf': args.pdf,
        'epub': args.epub,
        'mobi': args.mobi,
        'azw': args.azw,
        'azw3': args.azw3,
        'fb2': args.fb2,
    }
    
    if any(enabled_formats.values()):
        return {f'.{fmt}' for fmt, enabled in enabled_formats.items() if enabled}
    
    # Otherwise, process all except explicitly disabled
    disabled_formats = {
        'pdf': args.no_pdf,
        'epub': args.no_epub,
        'mobi': args.no_mobi,
        'azw': args.no_azw,
        'azw3': args.no_azw3,
        'fb2': args.no_fb2,
    }
    
    all_formats = {'.pdf', '.epub', '.mobi', '.azw', '.azw3', '.fb2'}
    return {f'.{fmt}' for fmt in all_formats if not disabled_formats.get(fmt, False)}


def rename_ebooks(
    directory: str,
    backup: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    template: str = "{title}",
    case_style: str = "original",
    max_length: int = 100,
    space_replacement: str = "none",
    use_ocr: bool = True,
    use_online_search: bool = False,
    online_prefer: bool = False,
    verbose: bool = True,
    formats: Optional[Set[str]] = None,
    log_file: Optional[str] = None,
    interactive: bool = True,
    replace_and_with_ampersand: bool = False  # NEW PARAMETER
) -> Tuple[int, int, int]:
    """Rename all supported ebook formats."""
    if log_file:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='w'
        )
    
    def log_print(message: str, color: str = ""):
        if verbose:
            reset = Colors.RESET if hasattr(Colors, 'RESET') and Colors.RESET else ''
            print(f"{color}{message}{reset}")
        if log_file:
            clean_message = re.sub(r'\033\[\d+m', '', message)
            logging.info(clean_message)
    
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"❌ '{directory}' is not a valid directory")
    
    # Determine which formats to process
    if formats is None:
        formats = {'.pdf', '.epub', '.mobi', '.azw', '.azw3', '.fb2'}
    
    # Build glob patterns for selected formats
    glob_patterns = [f'*{fmt}' for fmt in formats]
    
    ebook_files = []
    for pattern in glob_patterns:
        ebook_files.extend(path.glob(pattern))
    
    if not ebook_files:
        log_print(f"📭 No ebook files found in '{directory}' (looking for: {', '.join(formats)})", Colors.YELLOW)
        return 0, 0, 0
    
    backup_dir = path / "ebook_backups" if backup else None
    if backup_dir:
        backup_dir.mkdir(exist_ok=True)
        log_print(f"💾 Backups will be saved to: {backup_dir}", Colors.BLUE)
    
    success = 0
    errors = 0
    duplicates = []
    skipped = 0
    skip_all_duplicates = False
    
    format_count = {fmt: 0 for fmt in formats}
    for f in ebook_files:
        format_count[f.suffix.lower()] = format_count.get(f.suffix.lower(), 0) + 1
    
    counts_str = ", ".join([f"{count} {ext.upper()}" for ext, count in format_count.items() if count > 0])
    log_print(f"📊 Processing {len(ebook_files)} files ({counts_str})", Colors.BOLD)
    
    for ebook_file in ebook_files:
        try:
            original_name = ebook_file.name
            log_print(f"\n📄 Processing: {original_name}", Colors.BOLD)
            
            extractor = get_extractor(ebook_file, verbose=verbose, template=template, case_style=case_style,
                                    max_len=max_length, space_replacement=space_replacement, use_ocr=use_ocr,
                                    use_online_search=use_online_search, online_prefer=online_prefer,
                                    replace_and_with_ampersand=replace_and_with_ampersand)  # NEW PARAMETER PASSED
            
            metadata = extractor.extract_metadata(ebook_file)
            
            if not metadata['title']:
                log_print(f"  ⚠️ Could not extract title, using filename fallback", Colors.YELLOW)
                metadata['title'] = ebook_file.stem
            
            if metadata.get('_attempts'):
                for attempt in metadata['_attempts']:
                    log_print(f"  🔍 {attempt}", Colors.CYAN)
            
            info_parts = []
            if metadata['author']:
                info_parts.append(f"👤 {metadata['author']}")
            if metadata['year']:
                info_parts.append(f"📅 {metadata['year']}")
            if metadata['isbn']:
                info_parts.append(f"📖 {metadata['isbn']}")
            
            if info_parts:
                log_print(f"  ℹ️ Extracted: {', '.join(info_parts)}", Colors.BLUE)
            
            base_filename = extractor.format_filename(metadata)
            new_filename = f"{base_filename}{ebook_file.suffix}"
            new_filepath = ebook_file.parent / new_filename
            
            if new_filepath.exists() and not overwrite:
                duplicates.append((ebook_file, new_filename, original_name))
                log_print(f"  {Colors.RED}🚨 DUPLICATE: '{new_filename}' already exists!{Colors.RESET}", Colors.RED)
                
                if interactive and not skip_all_duplicates:
                    while True:
                        choice = input(f"  {Colors.YELLOW}[S]kip, [R]ename anyway, [A]ll duplicates, [Q]uit? {Colors.RESET}").strip().lower()
                        if choice == 's':
                            log_print(f"  ⏭️ Skipping {original_name}", Colors.YELLOW)
                            skipped += 1
                            success += 1
                            break
                        elif choice == 'r':
                            log_print(f"  ⚠️ Renaming anyway (will overwrite)", Colors.RED)
                            break
                        elif choice == 'a':
                            skip_all_duplicates = True
                            log_print(f"  ⏭️ Skipping all duplicates", Colors.YELLOW)
                            skipped += 1
                            success += 1
                            break
                        elif choice == 'q':
                            log_print(f"  🚫 Quitting...", Colors.RED)
                            return success, errors + len(ebook_files) - success, skipped
                        else:
                            log_print(f"  ❌ Invalid choice. Please try again.", Colors.RED)
                
                if skip_all_duplicates:
                    skipped += 1
                    success += 1
                    continue
                elif not interactive or not skip_all_duplicates:
                    continue
            
            if new_filepath == ebook_file:
                log_print(f"  ℹ️ Already correctly named", Colors.BLUE)
                success += 1
                continue
            
            if dry_run:
                log_print(f"  📋 Would rename to: {new_filename}", Colors.CYAN)
                success += 1
            else:
                if backup:
                    create_backup(ebook_file, backup_dir)
                
                ebook_file.rename(new_filepath)
                log_print(f"  ✅ Renamed to: {new_filename}", Colors.GREEN)
                success += 1
                
        except Exception as e:
            log_print(f"  ❌ Error: {e}", Colors.RED)
            errors += 1
    
    if duplicates:
        log_print(f"\n{Colors.YELLOW}📋 Duplicate Summary ({len(duplicates)} found):{Colors.RESET}", Colors.YELLOW)
        for original, new, orig_name in duplicates:
            log_print(f"  {Colors.RED}🚨 {orig_name} → {new}{Colors.RESET}", Colors.RED)
    
    return success, errors, skipped


def main():
    parser = argparse.ArgumentParser(
        description="📚 Universal Ebook Renamer v2.5 (PDF, EPUB, MOBI, FB2, AZW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📖 Supported Formats: PDF, EPUB, MOBI, AZW, AZW3, FB2

🎯 Format Selection Examples:
  # Process only PDF and EPUB files
  python ebook_rename.py "~/Books" --pdf --epub --template "{title} - {author}"

  # Process all except MOBI
  python ebook_rename.py "~/Books" --no-mobi --template "{title} ({year})"

  # Process only AZW3 files in quiet mode
  python ebook_rename.py "./inbox" --azw3 --quiet --template "[{isbn}] {title}"

🎯 Examples:
  # All formats, interactive mode (verbose by default)
  python ebook_rename.py "~/Books" --template "{title} - {author} ({year})"

  # Quiet mode (minimal output)
  python ebook_rename.py "~/Library" --quiet --template "{author}/{title}"

  # Explicitly enable verbose (default behavior)
  python ebook_rename.py "./inbox" --verbose --online-search --log-file "rename.log"
        """
    )
    
    parser.add_argument("directory", nargs="?", default=".", help="📁 Target directory")
    
    # 🔧 NEW: Format selection arguments
    format_group = parser.add_argument_group('Format Selection', 'Control which file types to process')
    format_group.add_argument("--pdf", action="store_true", help="📄 Process PDF files")
    format_group.add_argument("--no-pdf", action="store_true", help="🚫 Exclude PDF files")
    format_group.add_argument("--epub", action="store_true", help="📘 Process EPUB files")
    format_group.add_argument("--no-epub", action="store_true", help="🚫 Exclude EPUB files")
    format_group.add_argument("--mobi", action="store_true", help="📱 Process MOBI files")
    format_group.add_argument("--no-mobi", action="store_true", help="🚫 Exclude MOBI files")
    format_group.add_argument("--azw", action="store_true", help="🔥 Process AZW files")
    format_group.add_argument("--no-azw", action="store_true", help="🚫 Exclude AZW files")
    format_group.add_argument("--azw3", action="store_true", help="🔥 Process AZW3 files")
    format_group.add_argument("--no-azw3", action="store_true", help="🚫 Exclude AZW3 files")
    format_group.add_argument("--fb2", action="store_true", help="📚 Process FB2 files")
    format_group.add_argument("--no-fb2", action="store_true", help="🚫 Exclude FB2 files")
    
    parser.add_argument("-b", "--no-backup", action="store_true", help="💾 Skip backups")
    parser.add_argument("-n", "--dry-run", action="store_true", help="🔍 Preview only")
    parser.add_argument("-f", "--force", action="store_true", help="⚡ Overwrite duplicates")
    parser.add_argument("--template", default="{title}", help="🏷️ Template: {title}, {author}, {year}, {isbn}")
    parser.add_argument("--case", choices=["original", "lower", "upper", "title"], default="original", help="🔄 Case transformation")
    parser.add_argument("--max-length", type=int, default=100, help="📏 Max filename length")
    parser.add_argument("--replace-spaces", choices=["none", "underscore", "dash", "dot"], default="none", help="⬜ Space replacement")
    parser.add_argument("--no-ocr", action="store_true", help="🤖 Disable OCR correction")
    parser.add_argument("--online-search", action="store_true", help="🔍 Enable Google Books API cross-referencing")
    parser.add_argument("--online-prefer", action="store_true", help="🌐 Prefer online data over local when available")
    parser.add_argument("--verbose", action="store_true", default=True, help="🔊 Enable verbose output (default)")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="🔕 Disable verbose output")
    
    # NEW ARGUMENT
    parser.add_argument("--replace-and-with-ampersand", action="store_true", 
                        help="🔗 Replace ' and ' with ' & ' in titles")
    
    parser.add_argument("--log-file", type=str, help="📜 Log to file")
    parser.add_argument("--non-interactive", action="store_true", help="🎮 Auto-skip duplicates")
    parser.add_argument("--no-color", action="store_true", help="🎨 Disable colors")
    
    args = parser.parse_args()
    
    if args.no_color:
        Colors.disable()
    
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
        logging.info("🚀 Ebook Rename Tool Started")
    
    # 🔧 NEW: Determine formats to process
    formats_to_process = get_formats_to_process(args)
    
    if args.verbose:
        print("=" * 70)
        print("📚 Universal Ebook Renamer v2.5")
        print("=" * 70)
        print(f"📁 Directory: {args.directory}")
        print(f"🏷️ Template: '{args.template}'")
        print(f"🎨 Case: {args.case}")
        print(f"📏 Max length: {args.max_length}")
        print(f"⬜ Replace spaces: {args.replace_spaces}")
        print(f"🤖 OCR: {'Enabled' if not args.no_ocr else 'Disabled'}")
        print(f"🔍 Online Search: {'Enabled' if args.online_search else 'Disabled'}")
        print(f"🌐 Online Prefer: {'Yes' if args.online_prefer else 'No'}")
        print(f"🔗 Replace 'and' with '&': {'Yes' if args.replace_and_with_ampersand else 'No'}")  # NEW
        print(f"🎮 Interactive: {'Yes' if not args.non_interactive else 'No'}")
        print(f"📄 Formats: {', '.join(sorted(formats_to_process)) if formats_to_process else 'None'}")
        if args.log_file:
            print(f"📜 Logging: {args.log_file}")
        print("=" * 70)
    
    # Pass the new argument to rename_ebooks
    success, errors, skipped = rename_ebooks(
        directory=args.directory,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        overwrite=args.force,
        template=args.template,
        case_style=args.case,
        max_length=args.max_length,
        space_replacement=args.replace_spaces,
        use_ocr=not args.no_ocr,
        use_online_search=args.online_search,
        online_prefer=args.online_prefer,
        verbose=args.verbose,
        formats=formats_to_process,
        log_file=args.log_file,
        interactive=not args.non_interactive,
        replace_and_with_ampersand=args.replace_and_with_ampersand  # NEW
    )
    
    if args.verbose:
        print("=" * 70)
        print(f"{Colors.BOLD}📊 Summary: {success} renamed, {errors} errors, {skipped} skipped{Colors.RESET}")
        print("=" * 70)
    
    if args.log_file:
        logging.info(f"Summary: {success} renamed, {errors} errors, {skipped} skipped")
        print(f"\n📄 Log saved to: {args.log_file}")


if __name__ == "__main__":
    main()
