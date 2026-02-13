"""
Indicator Parser
Parses technical indicator values from OCR text
"""

import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaner import (
    clean_ocr_text,
    extract_numbers,
    extract_indicator_value,
    clean_indicator_text,
    parse_percentage
)


@dataclass
class IndicatorData:
    """Container for parsed indicator data."""
    price: Optional[float] = None
    rsi: Optional[float] = None
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume: Optional[float] = None
    volume_ma: Optional[float] = None
    change_percent: Optional[float] = None


class IndicatorParser:
    """
    Parser for extracting technical indicator values from OCR text.
    Handles various formats from different charting platforms.
    """
    
    def __init__(self):
        """Initialize indicator parser with pattern definitions."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for indicator extraction."""
        # Price patterns
        self.price_patterns = [
            re.compile(r'(?:price|last|close)[:\s]*\$?(\d+[,.]?\d*\.?\d*)', re.IGNORECASE),
            re.compile(r'\$(\d{1,6}[,.]?\d*\.?\d*)', re.IGNORECASE),
            re.compile(r'(\d{2,6}\.\d{2,8})', re.IGNORECASE),  # Crypto prices
        ]
        
        # RSI patterns
        self.rsi_patterns = [
            re.compile(r'RSI[(\d\s)]*[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'RSI\s*\(?\d*\)?[:\s]*(\d+\.?\d*)', re.IGNORECASE),
        ]
        
        # MACD patterns
        self.macd_patterns = [
            re.compile(r'MACD[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'MACD\s*Line[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
        ]
        
        self.macd_signal_patterns = [
            re.compile(r'Signal[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'MACD\s*Signal[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
        ]
        
        self.macd_hist_patterns = [
            re.compile(r'Hist(?:ogram)?[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'MACD\s*Hist[:\s]*(-?\d+\.?\d*)', re.IGNORECASE),
        ]
        
        # EMA patterns
        self.ema_patterns = {
            12: re.compile(r'EMA\s*\(?12\)?[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            26: re.compile(r'EMA\s*\(?26\)?[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            50: re.compile(r'EMA\s*\(?50\)?[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            200: re.compile(r'EMA\s*\(?200\)?[:\s]*(\d+\.?\d*)', re.IGNORECASE),
        }
        
        # Bollinger Bands patterns
        self.bb_upper_patterns = [
            re.compile(r'BB\s*Upper[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'Upper\s*Band[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'Bollinger.*Upper[:\s]*(\d+\.?\d*)', re.IGNORECASE),
        ]
        
        self.bb_middle_patterns = [
            re.compile(r'BB\s*Middle[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'Middle\s*Band[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'BB\s*SMA[:\s]*(\d+\.?\d*)', re.IGNORECASE),
        ]
        
        self.bb_lower_patterns = [
            re.compile(r'BB\s*Lower[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'Lower\s*Band[:\s]*(\d+\.?\d*)', re.IGNORECASE),
            re.compile(r'Bollinger.*Lower[:\s]*(\d+\.?\d*)', re.IGNORECASE),
        ]
        
        # Volume patterns
        self.volume_patterns = [
            re.compile(r'Vol(?:ume)?[:\s]*(\d+\.?\d*)\s*([KMBkmb])?', re.IGNORECASE),
            re.compile(r'V[:\s]*(\d+\.?\d*)\s*([KMBkmb])?', re.IGNORECASE),
        ]
    
    def parse(self, text: str) -> IndicatorData:
        """
        Parse all technical indicators from OCR text.
        
        Args:
            text: OCR extracted text
            
        Returns:
            IndicatorData object with parsed values
        """
        # Clean the input text
        cleaned_text = clean_indicator_text(text)
        
        data = IndicatorData()
        
        # Parse each indicator
        data.price = self._parse_price(cleaned_text)
        data.rsi = self._parse_rsi(cleaned_text)
        
        macd_data = self._parse_macd(cleaned_text)
        data.macd_value = macd_data.get('value')
        data.macd_signal = macd_data.get('signal')
        data.macd_histogram = macd_data.get('histogram')
        
        ema_data = self._parse_ema(cleaned_text)
        data.ema_12 = ema_data.get(12)
        data.ema_26 = ema_data.get(26)
        data.ema_50 = ema_data.get(50)
        data.ema_200 = ema_data.get(200)
        
        bb_data = self._parse_bollinger(cleaned_text)
        data.bollinger_upper = bb_data.get('upper')
        data.bollinger_middle = bb_data.get('middle')
        data.bollinger_lower = bb_data.get('lower')
        
        volume_data = self._parse_volume(cleaned_text)
        data.volume = volume_data.get('volume')
        data.volume_ma = volume_data.get('volume_ma')
        
        data.change_percent = self._parse_change(cleaned_text)
        
        return data
    
    def _parse_price(self, text: str) -> Optional[float]:
        """Parse price from text."""
        for pattern in self.price_patterns:
            match = pattern.search(text)
            if match:
                try:
                    price_str = match.group(1).replace(',', '')
                    return float(price_str)
                except ValueError:
                    continue
        return None
    
    def _parse_rsi(self, text: str) -> Optional[float]:
        """Parse RSI value from text."""
        for pattern in self.rsi_patterns:
            match = pattern.search(text)
            if match:
                try:
                    rsi = float(match.group(1))
                    if 0 <= rsi <= 100:  # Validate RSI range
                        return rsi
                except ValueError:
                    continue
        return None
    
    def _parse_macd(self, text: str) -> Dict[str, Optional[float]]:
        """Parse MACD values from text."""
        result = {'value': None, 'signal': None, 'histogram': None}
        
        for pattern in self.macd_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['value'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        for pattern in self.macd_signal_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['signal'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        for pattern in self.macd_hist_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['histogram'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return result
    
    def _parse_ema(self, text: str) -> Dict[int, Optional[float]]:
        """Parse EMA values from text."""
        result = {}
        
        for period, pattern in self.ema_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    result[period] = float(match.group(1))
                except ValueError:
                    continue
        
        return result
    
    def _parse_bollinger(self, text: str) -> Dict[str, Optional[float]]:
        """Parse Bollinger Bands values from text."""
        result = {'upper': None, 'middle': None, 'lower': None}
        
        for pattern in self.bb_upper_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['upper'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        for pattern in self.bb_middle_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['middle'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        for pattern in self.bb_lower_patterns:
            match = pattern.search(text)
            if match:
                try:
                    result['lower'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        return result
    
    def _parse_volume(self, text: str) -> Dict[str, Optional[float]]:
        """Parse volume values from text."""
        result = {'volume': None, 'volume_ma': None}
        
        for pattern in self.volume_patterns:
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group(1))
                    suffix = match.group(2) if match.lastindex >= 2 else None
                    
                    if suffix:
                        multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
                        value *= multipliers.get(suffix.lower(), 1)
                    
                    if result['volume'] is None:
                        result['volume'] = value
                    else:
                        result['volume_ma'] = value
                        break
                except ValueError:
                    continue
        
        return result
    
    def _parse_change(self, text: str) -> Optional[float]:
        """Parse percentage change from text."""
        return parse_percentage(text)
    
    def to_dict(self, data: IndicatorData) -> Dict:
        """
        Convert IndicatorData to dictionary.
        
        Args:
            data: IndicatorData object
            
        Returns:
            Dictionary representation
        """
        return {
            'price': data.price,
            'rsi': data.rsi,
            'macd': {
                'value': data.macd_value,
                'signal': data.macd_signal,
                'histogram': data.macd_histogram,
            },
            'ema': {
                '12': data.ema_12,
                '26': data.ema_26,
                '50': data.ema_50,
                '200': data.ema_200,
            },
            'bollinger_bands': {
                'upper': data.bollinger_upper,
                'middle': data.bollinger_middle,
                'lower': data.bollinger_lower,
            },
            'volume': {
                'current': data.volume,
                'ma': data.volume_ma,
            },
            'change_percent': data.change_percent,
        }


def parse_indicators(text: str) -> Dict:
    """
    Convenience function to parse indicators from text.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Dictionary with parsed indicator values
    """
    parser = IndicatorParser()
    data = parser.parse(text)
    return parser.to_dict(data)
