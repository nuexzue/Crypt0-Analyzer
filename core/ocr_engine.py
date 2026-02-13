"""
OCR Engine
Handles text extraction from images using pytesseract
"""

import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Optional, List, Union
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import (
    load_image, 
    preprocess_for_ocr, 
    resize_for_ocr,
    enhance_numbers
)


class OCREngine:
    """
    OCR Engine for extracting text from trading screenshots.
    Optimized for dark-themed trading interfaces.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR Engine.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Default OCR configuration
        self.config = {
            'standard': '--oem 3 --psm 6',
            'single_line': '--oem 3 --psm 7',
            'numbers_only': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-,',
            'sparse': '--oem 3 --psm 11',
        }
    
    def extract_text(
        self, 
        image: Union[str, bytes, np.ndarray, Image.Image],
        preprocess_mode: str = "standard",
        ocr_mode: str = "standard"
    ) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Image source (path, bytes, array, or PIL Image)
            preprocess_mode: Preprocessing mode ('standard', 'dark_theme', 'orderbook')
            ocr_mode: OCR configuration mode
            
        Returns:
            Extracted text
        """
        # Load and preprocess image
        img_array = load_image(image)
        
        # Resize for better OCR accuracy
        img_array = resize_for_ocr(img_array)
        
        # Preprocess
        processed = preprocess_for_ocr(img_array, mode=preprocess_mode)
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(processed)
        
        # Get OCR config
        config = self.config.get(ocr_mode, self.config['standard'])
        
        # Extract text
        try:
            text = pytesseract.image_to_string(pil_image, config=config)
            return text
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {str(e)}")
    
    def extract_with_confidence(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        preprocess_mode: str = "standard"
    ) -> List[Dict]:
        """
        Extract text with confidence scores and bounding boxes.
        
        Args:
            image: Image source
            preprocess_mode: Preprocessing mode
            
        Returns:
            List of dictionaries with text, confidence, and bbox
        """
        img_array = load_image(image)
        img_array = resize_for_ocr(img_array)
        processed = preprocess_for_ocr(img_array, mode=preprocess_mode)
        pil_image = Image.fromarray(processed)
        
        try:
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {str(e)}")
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 0:
                results.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return results
    
    def extract_numbers_from_region(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        region: tuple = None
    ) -> str:
        """
        Extract numbers from a specific region of the image.
        
        Args:
            image: Image source
            region: Tuple (x, y, width, height) or None for full image
            
        Returns:
            Extracted text (numbers focused)
        """
        img_array = load_image(image)
        
        if region:
            x, y, w, h = region
            img_array = img_array[y:y+h, x:x+w]
        
        # Enhance for number recognition
        enhanced = enhance_numbers(img_array)
        pil_image = Image.fromarray(enhanced)
        
        config = self.config['numbers_only']
        
        try:
            text = pytesseract.image_to_string(pil_image, config=config)
            return text
        except Exception as e:
            raise RuntimeError(f"Number extraction failed: {str(e)}")
    
    def extract_chart_text(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image]
    ) -> str:
        """
        Extract text from a trading chart screenshot.
        Optimized for dark-themed charts.
        
        Args:
            image: Chart image
            
        Returns:
            Extracted text
        """
        return self.extract_text(image, preprocess_mode="dark_theme", ocr_mode="standard")
    
    def extract_orderbook_text(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image]
    ) -> str:
        """
        Extract text from an orderbook screenshot.
        
        Args:
            image: Orderbook image
            
        Returns:
            Extracted text
        """
        return self.extract_text(image, preprocess_mode="orderbook", ocr_mode="standard")
    
    def get_structured_data(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        preprocess_mode: str = "dark_theme"
    ) -> Dict:
        """
        Get structured OCR data including line-by-line extraction.
        
        Args:
            image: Image source
            preprocess_mode: Preprocessing mode
            
        Returns:
            Dictionary with raw text and structured line data
        """
        img_array = load_image(image)
        img_array = resize_for_ocr(img_array)
        processed = preprocess_for_ocr(img_array, mode=preprocess_mode)
        pil_image = Image.fromarray(processed)
        
        try:
            # Get full text
            raw_text = pytesseract.image_to_string(pil_image, config=self.config['standard'])
            
            # Get detailed data
            detailed = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {str(e)}")
        
        # Organize by lines
        lines = []
        current_line = []
        current_line_num = -1
        
        for i in range(len(detailed['text'])):
            if detailed['text'][i].strip():
                line_num = detailed['line_num'][i]
                
                if line_num != current_line_num and current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                
                current_line.append(detailed['text'][i])
                current_line_num = line_num
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return {
            'raw_text': raw_text,
            'lines': lines,
            'word_count': len([t for t in detailed['text'] if t.strip()]),
        }


# Singleton instance for convenience
_engine_instance = None


def get_ocr_engine(tesseract_cmd: Optional[str] = None) -> OCREngine:
    """Get or create OCR engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OCREngine(tesseract_cmd)
    return _engine_instance
