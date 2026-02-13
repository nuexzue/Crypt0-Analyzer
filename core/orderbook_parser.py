"""
Orderbook Parser
Parses orderbook data from OCR text
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaner import (
    clean_ocr_text,
    extract_numbers,
    split_by_lines,
    extract_bid_ask_sections,
    normalize_number_string
)


@dataclass
class OrderbookEntry:
    """Single orderbook entry."""
    price: float
    volume: float
    total: Optional[float] = None


@dataclass 
class OrderbookData:
    """Container for parsed orderbook data."""
    bids: List[OrderbookEntry] = field(default_factory=list)
    asks: List[OrderbookEntry] = field(default_factory=list)
    spread: Optional[float] = None
    mid_price: Optional[float] = None


class OrderbookParser:
    """
    Parser for extracting orderbook data from screenshots.
    Handles Binance and similar exchange formats.
    """
    
    def __init__(self):
        """Initialize orderbook parser."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for orderbook parsing."""
        # Pattern for price-volume pairs
        self.entry_pattern = re.compile(
            r'(\d+\.?\d*)\s+(\d+\.?\d*)\s*(\d+\.?\d*)?',
            re.IGNORECASE
        )
        
        # Pattern for numbers with K/M suffixes
        self.volume_suffix_pattern = re.compile(
            r'(\d+\.?\d*)\s*([KMBkmb])?',
            re.IGNORECASE
        )
        
        # Spread pattern
        self.spread_pattern = re.compile(
            r'spread[:\s]*(\d+\.?\d*)',
            re.IGNORECASE
        )
    
    def parse(self, text: str) -> OrderbookData:
        """
        Parse orderbook data from OCR text.
        
        Args:
            text: OCR extracted text from orderbook screenshot
            
        Returns:
            OrderbookData object with parsed bid/ask data
        """
        data = OrderbookData()
        
        # Clean the text
        cleaned = clean_ocr_text(text)
        
        # Try to separate bid and ask sections
        bid_text, ask_text = extract_bid_ask_sections(cleaned)
        
        # Parse bids
        data.bids = self._parse_entries(bid_text, is_bid=True)
        
        # Parse asks  
        data.asks = self._parse_entries(ask_text, is_bid=False)
        
        # Calculate spread and mid price
        if data.bids and data.asks:
            best_bid = max(entry.price for entry in data.bids)
            best_ask = min(entry.price for entry in data.asks)
            
            if best_ask > best_bid:
                data.spread = best_ask - best_bid
                data.mid_price = (best_bid + best_ask) / 2
        
        return data
    
    def parse_binance_format(self, text: str) -> OrderbookData:
        """
        Parse Binance-specific orderbook format.
        
        Args:
            text: OCR text from Binance orderbook
            
        Returns:
            OrderbookData object
        """
        data = OrderbookData()
        lines = split_by_lines(text)
        
        # Binance orderbook typically has asks on top, bids on bottom
        # Find the middle/spread line
        spread_idx = None
        for i, line in enumerate(lines):
            if 'spread' in line.lower() or self._is_separator_line(line):
                spread_idx = i
                break
        
        if spread_idx:
            ask_lines = lines[:spread_idx]
            bid_lines = lines[spread_idx + 1:]
        else:
            # Split in half if no clear separator
            mid = len(lines) // 2
            ask_lines = lines[:mid]
            bid_lines = lines[mid:]
        
        # Parse ask entries (reverse order - lowest ask first)
        for line in reversed(ask_lines):
            entry = self._parse_line(line)
            if entry:
                data.asks.append(entry)
        
        # Parse bid entries
        for line in bid_lines:
            entry = self._parse_line(line)
            if entry:
                data.bids.append(entry)
        
        # Calculate spread
        if data.bids and data.asks:
            best_bid = max(entry.price for entry in data.bids)
            best_ask = min(entry.price for entry in data.asks)
            
            if best_ask > best_bid:
                data.spread = best_ask - best_bid
                data.mid_price = (best_bid + best_ask) / 2
        
        return data
    
    def _parse_entries(self, text: str, is_bid: bool = True) -> List[OrderbookEntry]:
        """
        Parse orderbook entries from text section.
        
        Args:
            text: Text section containing entries
            is_bid: Whether parsing bid or ask side
            
        Returns:
            List of OrderbookEntry objects
        """
        entries = []
        lines = split_by_lines(text)
        
        for line in lines:
            entry = self._parse_line(line)
            if entry:
                entries.append(entry)
        
        # Sort appropriately
        if is_bid:
            entries.sort(key=lambda x: x.price, reverse=True)  # Highest bid first
        else:
            entries.sort(key=lambda x: x.price)  # Lowest ask first
        
        return entries
    
    def _parse_line(self, line: str) -> Optional[OrderbookEntry]:
        """
        Parse a single orderbook line.
        
        Args:
            line: Single line of orderbook text
            
        Returns:
            OrderbookEntry or None
        """
        # Extract all numbers from line
        numbers = extract_numbers(line)
        
        if len(numbers) >= 2:
            # Determine which is price and which is volume
            # Usually price has more decimal places for crypto
            price = numbers[0]
            volume = numbers[1]
            total = numbers[2] if len(numbers) >= 3 else None
            
            # Basic validation
            if price > 0 and volume > 0:
                # Handle volume suffixes
                volume = self._parse_volume_with_suffix(line, volume)
                
                return OrderbookEntry(
                    price=price,
                    volume=volume,
                    total=total
                )
        
        return None
    
    def _parse_volume_with_suffix(self, line: str, default_volume: float) -> float:
        """Parse volume considering K/M/B suffixes."""
        match = self.volume_suffix_pattern.search(line)
        if match:
            value = float(match.group(1))
            suffix = match.group(2)
            
            if suffix:
                multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
                value *= multipliers.get(suffix.lower(), 1)
                return value
        
        return default_volume
    
    def _is_separator_line(self, line: str) -> bool:
        """Check if line is a separator between bid/ask."""
        separators = ['---', '===', '***', 'spread', 'last', 'price']
        line_lower = line.lower()
        
        for sep in separators:
            if sep in line_lower:
                return True
        
        # Check if line has very few alphanumeric characters
        alnum_count = sum(1 for c in line if c.isalnum())
        return alnum_count < 3
    
    def to_dict(self, data: OrderbookData) -> Dict:
        """
        Convert OrderbookData to dictionary.
        
        Args:
            data: OrderbookData object
            
        Returns:
            Dictionary representation
        """
        return {
            'bids': [
                {'price': e.price, 'volume': e.volume, 'total': e.total}
                for e in data.bids
            ],
            'asks': [
                {'price': e.price, 'volume': e.volume, 'total': e.total}
                for e in data.asks
            ],
            'spread': data.spread,
            'mid_price': data.mid_price,
            'bid_count': len(data.bids),
            'ask_count': len(data.asks),
        }
    
    def get_summary(self, data: OrderbookData) -> Dict:
        """
        Get orderbook summary statistics.
        
        Args:
            data: OrderbookData object
            
        Returns:
            Dictionary with summary statistics
        """
        total_bid_volume = sum(e.volume for e in data.bids) if data.bids else 0
        total_ask_volume = sum(e.volume for e in data.asks) if data.asks else 0
        
        best_bid = max((e.price for e in data.bids), default=0)
        best_ask = min((e.price for e in data.asks), default=0)
        
        return {
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': data.spread,
            'mid_price': data.mid_price,
            'bid_count': len(data.bids),
            'ask_count': len(data.asks),
        }


def parse_orderbook(text: str) -> Dict:
    """
    Convenience function to parse orderbook from text.
    
    Args:
        text: OCR extracted text
        
    Returns:
        Dictionary with parsed orderbook data
    """
    parser = OrderbookParser()
    data = parser.parse(text)
    return parser.to_dict(data)
