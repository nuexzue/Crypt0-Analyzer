"""
Image Processing Utilities
Handles image preprocessing for OCR optimization
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional


def load_image(image_source: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load image from various sources and convert to numpy array.
    
    Args:
        image_source: Path string, bytes, numpy array, or PIL Image
        
    Returns:
        numpy array in BGR format
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_source}")
        return img
    elif isinstance(image_source, bytes):
        nparr = np.frombuffer(image_source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return img
    elif isinstance(image_source, np.ndarray):
        return image_source
    elif isinstance(image_source, Image.Image):
        return cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")


def preprocess_for_ocr(image: np.ndarray, mode: str = "standard") -> np.ndarray:
    """
    Preprocess image for optimal OCR results.
    
    Args:
        image: Input image as numpy array
        mode: Preprocessing mode - 'standard', 'dark_theme', 'orderbook'
        
    Returns:
        Preprocessed image optimized for OCR
    """
    if mode == "dark_theme":
        return _preprocess_dark_theme(image)
    elif mode == "orderbook":
        return _preprocess_orderbook(image)
    else:
        return _preprocess_standard(image)


def _preprocess_standard(image: np.ndarray) -> np.ndarray:
    """Standard preprocessing pipeline."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Slight dilation to connect text components
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    
    return processed


def _preprocess_dark_theme(image: np.ndarray) -> np.ndarray:
    """Preprocessing optimized for dark trading chart themes."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert colors (dark backgrounds become light)
    inverted = cv2.bitwise_not(gray)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def _preprocess_orderbook(image: np.ndarray) -> np.ndarray:
    """Preprocessing optimized for orderbook screenshots."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if dark theme (mean value < 127)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Binary threshold
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def resize_for_ocr(image: np.ndarray, target_height: int = 2000) -> np.ndarray:
    """
    Resize image to optimal size for OCR while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if h >= target_height:
        return image
    
    scale = target_height / h
    new_width = int(w * scale)
    
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized


def extract_color_regions(image: np.ndarray, color: str) -> np.ndarray:
    """
    Extract regions of specific colors (useful for bid/ask separation).
    
    Args:
        image: Input BGR image
        color: 'green', 'red', 'blue', or 'white'
        
    Returns:
        Binary mask of color regions
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'green': ((35, 50, 50), (85, 255, 255)),
        'red': ((0, 50, 50), (10, 255, 255)),  # Also check upper red range
        'blue': ((100, 50, 50), (130, 255, 255)),
        'white': ((0, 0, 200), (180, 30, 255)),
    }
    
    if color not in color_ranges:
        raise ValueError(f"Unsupported color: {color}")
    
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # For red, also check the upper range
    if color == 'red':
        lower2 = (170, 50, 50)
        upper2 = (180, 255, 255)
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)
    
    return mask


def crop_region(image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a specific region from the image.
    
    Args:
        image: Input image
        region: Tuple of (x, y, width, height)
        
    Returns:
        Cropped image region
    """
    x, y, w, h = region
    return image[y:y+h, x:x+w].copy()


def detect_text_regions(image: np.ndarray) -> list:
    """
    Detect potential text regions in the image using contour detection.
    
    Args:
        image: Preprocessed binary image
        
    Returns:
        List of bounding boxes (x, y, w, h) for text regions
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter by aspect ratio and size
        aspect_ratio = w / h if h > 0 else 0
        if 0.1 < aspect_ratio < 20 and w > 10 and h > 5:
            regions.append((x, y, w, h))
    
    # Sort by y position then x position
    regions.sort(key=lambda r: (r[1], r[0]))
    
    return regions


def enhance_numbers(image: np.ndarray) -> np.ndarray:
    """
    Apply enhancement specifically for number recognition.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
    
    # Sharpen
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(morph, -1, sharpen_kernel)
    
    return sharpened
