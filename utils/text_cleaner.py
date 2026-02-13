"""
Text Cleaning Utilities
Cleans and normalizes extracted OCR text for parsing
"""

import re
from typing import Optional, List, Tuple


def clean_ocr_text(text: str) -> str:
    """
    Clean raw OCR output text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace common OCR misreadings
    replacements = {
        '|': '1',
        'l': '1',
        'O': '0',
        'o': '0',
        'S': '5',
        'B': '8',
        'g': '9',
        'Z': '2',
        '—': '-',
        '–': '-',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
    }
    
    cleaned = text
    for old, new in replacements.items():
        # Only replace in number contexts
        cleaned = re.sub(rf'(?<=\d){re.escape(old)}(?=\d)', new, cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric values from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    # Pattern for numbers including decimals and negatives
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            num = float(match)
            numbers.append(num)
        except ValueError:
            continue
    
    return numbers


def extract_price(text: str) -> Optional[float]:
    """
    Extract price value from text (handles common formats).
    
    Args:
        text: Text containing price
        
    Returns:
        Extracted price or None
    """
    # Common price patterns
    patterns = [
        r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,8})?)',  # $1,234.56 or 1234.56
        r'(\d+\.?\d*)\s*(?:USD|USDT|BTC|ETH)',  # 1234.56 USDT
        r'Price[:\s]*(\d+\.?\d*)',  # Price: 1234.56
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                price_str = match.group(1).replace(',', '')
                return float(price_str)
            except ValueError:
                continue
    
    return None


def extract_indicator_value(text: str, indicator: str) -> Optional[float]:
    """
    Extract a specific indicator value from text.
    
    Args:
        text: Text containing indicator data
        indicator: Indicator name (RSI, MACD, EMA, etc.)
        
    Returns:
        Extracted value or None
    """
    # Build pattern for the indicator
    patterns = [
        rf'{indicator}[:\s]*(-?\d+\.?\d*)',
        rf'{indicator}\s*\(.*?\)[:\s]*(-?\d+\.?\d*)',
        rf'{indicator}.*?(-?\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None


def extract_volume(text: str) -> Optional[float]:
    """
    Extract volume value from text (handles K, M, B suffixes).
    
    Args:
        text: Text containing volume
        
    Returns:
        Extracted volume as float
    """
    # Pattern for volumes with suffixes
    pattern = r'(?:Vol(?:ume)?[:\s]*)?(\d+\.?\d*)\s*([KMBkmb])?'
    
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            value = float(match.group(1))
            suffix = match.group(2)
            
            if suffix:
                multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
                value *= multipliers.get(suffix.lower(), 1)
            
            return value
        except ValueError:
            return None
    
    return None


def extract_orderbook_row(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract price and volume from an orderbook row.
    
    Args:
        text: Single row text from orderbook
        
    Returns:
        Tuple of (price, volume) or None
    """
    # Clean the text
    cleaned = clean_ocr_text(text)
    
    # Extract numbers
    numbers = extract_numbers(cleaned)
    
    if len(numbers) >= 2:
        # Usually price is first, volume is second (or vice versa)
        # Assume larger decimal precision is price for crypto
        price = numbers[0]
        volume = numbers[1]
        
        if price > 0 and volume > 0:
            return (price, volume)
    
    return None


def parse_percentage(text: str) -> Optional[float]:
    """
    Parse percentage value from text.
    
    Args:
        text: Text containing percentage
        
    Returns:
        Percentage as float (e.g., 5.5 for 5.5%)
    """
    pattern = r'(-?\d+\.?\d*)\s*%'
    match = re.search(pattern, text)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    return None


def normalize_number_string(text: str) -> str:
    """
    Normalize number string by removing common formatting.
    
    Args:
        text: Number string with potential formatting
        
    Returns:
        Normalized number string
    """
    # Remove currency symbols
    normalized = re.sub(r'[$€£¥₿]', '', text)
    
    # Remove spaces
    normalized = normalized.replace(' ', '')
    
    # Handle European number format (1.234,56 -> 1234.56)
    if ',' in normalized and '.' in normalized:
        if normalized.rindex(',') > normalized.rindex('.'):
            # European format
            normalized = normalized.replace('.', '').replace(',', '.')
        else:
            # US format
            normalized = normalized.replace(',', '')
    elif ',' in normalized:
        # Could be thousand separator or decimal
        parts = normalized.split(',')
        if len(parts) == 2 and len(parts[1]) == 2:
            # Likely European decimal
            normalized = normalized.replace(',', '.')
        else:
            # Likely thousand separator
            normalized = normalized.replace(',', '')
    
    return normalized


def split_by_lines(text: str) -> List[str]:
    """
    Split text into lines and clean each line.
    
    Args:
        text: Multi-line text
        
    Returns:
        List of cleaned non-empty lines
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    
    return cleaned_lines


def extract_bid_ask_sections(text: str) -> Tuple[str, str]:
    """
    Split orderbook text into bid and ask sections.
    
    Args:
        text: Full orderbook text
        
    Returns:
        Tuple of (bid_text, ask_text)
    """
    text_lower = text.lower()
    
    # Try to find section markers
    bid_markers = ['bid', 'buy', 'bids']
    ask_markers = ['ask', 'sell', 'asks', 'offer']
    
    bid_start = -1
    ask_start = -1
    
    for marker in bid_markers:
        idx = text_lower.find(marker)
        if idx != -1:
            bid_start = idx
            break
    
    for marker in ask_markers:
        idx = text_lower.find(marker)
        if idx != -1:
            ask_start = idx
            break
    
    if bid_start != -1 and ask_start != -1:
        if bid_start < ask_start:
            bid_text = text[bid_start:ask_start]
            ask_text = text[ask_start:]
        else:
            ask_text = text[ask_start:bid_start]
            bid_text = text[bid_start:]
    else:
        # No clear markers, split in half
        mid = len(text) // 2
        bid_text = text[:mid]
        ask_text = text[mid:]
    
    return bid_text, ask_text


def clean_indicator_text(text: str) -> str:
    """
    Clean text specifically for indicator parsing.
    
    Args:
        text: Raw indicator text
        
    Returns:
        Cleaned text
    """
    # Remove common chart noise
    noise_patterns = [
        r'TradingView',
        r'Binance',
        r'[A-Z]{3,4}/[A-Z]{3,4}',  # Trading pairs
        r'©.*',
        r'www\..*',
    ]
    
    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()
