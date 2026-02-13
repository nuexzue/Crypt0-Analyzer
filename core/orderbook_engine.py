"""
Orderbook Analysis Engine
Analyzes orderbook data and produces weighted scores
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .orderbook_parser import OrderbookData, OrderbookEntry, OrderbookParser


@dataclass
class OrderbookScore:
    """Container for orderbook analysis scores."""
    overall_score: float  # -10 to +10
    imbalance_score: float
    spread_score: float
    wall_score: float
    depth_score: float
    signal: str  # 'buy', 'sell', 'neutral'
    confidence: float  # 0 to 1
    metrics: Dict
    details: Dict


class OrderbookEngine:
    """
    Orderbook analysis engine that computes scores
    based on bid/ask data, walls, and imbalances.
    """
    
    WALL_THRESHOLD_MULTIPLIER = 3.0  # Volume must be 3x average to be considered a wall
    
    def __init__(self):
        """Initialize orderbook engine."""
        self.parser = OrderbookParser()
    
    def analyze(self, orderbook_data: OrderbookData) -> OrderbookScore:
        """
        Perform orderbook analysis.
        
        Args:
            orderbook_data: Parsed orderbook data
            
        Returns:
            OrderbookScore with analysis results
        """
        # Calculate metrics
        metrics = self._calculate_metrics(orderbook_data)
        
        # Calculate individual scores
        imbalance_score = self._score_imbalance(metrics)
        spread_score = self._score_spread(metrics)
        wall_score = self._score_walls(metrics)
        depth_score = self._score_depth(metrics)
        
        # Calculate weighted overall score
        scores = {
            'imbalance': (imbalance_score, 0.35),
            'walls': (wall_score, 0.30),
            'spread': (spread_score, 0.20),
            'depth': (depth_score, 0.15),
        }
        
        overall_score = sum(
            score * weight 
            for score, weight in scores.values() 
            if score is not None
        )
        
        # Determine signal and confidence
        signal, confidence = self._determine_signal(overall_score, metrics)
        
        # Build details
        details = {
            'component_scores': {
                'imbalance': imbalance_score,
                'spread': spread_score,
                'walls': wall_score,
                'depth': depth_score,
            },
            'bid_count': len(orderbook_data.bids),
            'ask_count': len(orderbook_data.asks),
        }
        
        return OrderbookScore(
            overall_score=round(overall_score, 2),
            imbalance_score=imbalance_score or 0,
            spread_score=spread_score or 0,
            wall_score=wall_score or 0,
            depth_score=depth_score or 0,
            signal=signal,
            confidence=confidence,
            metrics=metrics,
            details=details
        )
    
    def analyze_from_text(self, ocr_text: str) -> OrderbookScore:
        """
        Analyze orderbook from OCR text.
        
        Args:
            ocr_text: Raw OCR text from orderbook screenshot
            
        Returns:
            OrderbookScore
        """
        orderbook_data = self.parser.parse(ocr_text)
        return self.analyze(orderbook_data)
    
    def _calculate_metrics(self, data: OrderbookData) -> Dict:
        """
        Calculate orderbook metrics.
        
        Args:
            data: OrderbookData object
            
        Returns:
            Dictionary of metrics
        """
        # Total volumes
        total_bid_volume = sum(e.volume for e in data.bids) if data.bids else 0
        total_ask_volume = sum(e.volume for e in data.asks) if data.asks else 0
        
        # Imbalance ratio
        total_volume = total_bid_volume + total_ask_volume
        if total_volume > 0:
            imbalance_ratio = (total_bid_volume - total_ask_volume) / total_volume
        else:
            imbalance_ratio = 0
        
        # Best bid/ask
        best_bid = max((e.price for e in data.bids), default=0)
        best_ask = min((e.price for e in data.asks), default=0)
        
        # Spread
        spread = data.spread or (best_ask - best_bid if best_ask > best_bid else 0)
        spread_percent = (spread / best_bid * 100) if best_bid > 0 else 0
        
        # Detect walls
        buy_walls = self._detect_walls(data.bids, True)
        sell_walls = self._detect_walls(data.asks, False)
        
        # Depth analysis (volume at different price levels)
        bid_depth_5pct = self._calculate_depth(data.bids, best_bid, 0.05, True)
        ask_depth_5pct = self._calculate_depth(data.asks, best_ask, 0.05, False)
        
        return {
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'imbalance_ratio': imbalance_ratio,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_percent': spread_percent,
            'mid_price': data.mid_price or ((best_bid + best_ask) / 2 if best_bid and best_ask else 0),
            'buy_walls': buy_walls,
            'sell_walls': sell_walls,
            'bid_depth_5pct': bid_depth_5pct,
            'ask_depth_5pct': ask_depth_5pct,
            'has_buy_wall': len(buy_walls) > 0,
            'has_sell_wall': len(sell_walls) > 0,
        }
    
    def _detect_walls(
        self, 
        entries: List[OrderbookEntry], 
        is_bid: bool
    ) -> List[Dict]:
        """
        Detect buy/sell walls in orderbook.
        
        Args:
            entries: List of orderbook entries
            is_bid: Whether analyzing bid side
            
        Returns:
            List of detected walls with price and volume
        """
        if not entries or len(entries) < 3:
            return []
        
        volumes = [e.volume for e in entries]
        avg_volume = sum(volumes) / len(volumes)
        threshold = avg_volume * self.WALL_THRESHOLD_MULTIPLIER
        
        walls = []
        for entry in entries:
            if entry.volume >= threshold:
                walls.append({
                    'price': entry.price,
                    'volume': entry.volume,
                    'volume_ratio': entry.volume / avg_volume,
                    'type': 'buy_wall' if is_bid else 'sell_wall'
                })
        
        return walls
    
    def _calculate_depth(
        self,
        entries: List[OrderbookEntry],
        reference_price: float,
        percent_range: float,
        is_bid: bool
    ) -> float:
        """
        Calculate total volume within percentage range of reference price.
        
        Args:
            entries: Orderbook entries
            reference_price: Reference price (best bid/ask)
            percent_range: Percentage range to consider
            is_bid: Whether bid side
            
        Returns:
            Total volume within range
        """
        if not entries or reference_price <= 0:
            return 0
        
        total = 0
        for entry in entries:
            price_diff_pct = abs(entry.price - reference_price) / reference_price
            if price_diff_pct <= percent_range:
                total += entry.volume
        
        return total
    
    def _score_imbalance(self, metrics: Dict) -> float:
        """
        Score based on bid/ask imbalance (-10 to +10).
        
        Positive imbalance (more bids): Bullish
        Negative imbalance (more asks): Bearish
        """
        ratio = metrics.get('imbalance_ratio', 0)
        
        # Scale ratio to -10 to +10
        # Ratio of +/-0.5 is considered extreme
        score = ratio * 20
        
        return max(min(score, 10), -10)
    
    def _score_spread(self, metrics: Dict) -> float:
        """
        Score based on spread (-10 to +10).
        
        Tight spread: Better liquidity (neutral to positive)
        Wide spread: Poor liquidity (negative)
        """
        spread_pct = metrics.get('spread_percent', 0)
        
        if spread_pct <= 0.01:
            return 5.0  # Very tight spread - excellent liquidity
        elif spread_pct <= 0.05:
            return 3.0  # Tight spread
        elif spread_pct <= 0.1:
            return 0.0  # Normal spread
        elif spread_pct <= 0.5:
            return -3.0  # Wide spread
        else:
            return -7.0  # Very wide spread - poor liquidity
    
    def _score_walls(self, metrics: Dict) -> float:
        """
        Score based on buy/sell walls (-10 to +10).
        
        Buy walls: Support, bullish
        Sell walls: Resistance, bearish
        """
        buy_walls = metrics.get('buy_walls', [])
        sell_walls = metrics.get('sell_walls', [])
        
        score = 0.0
        
        # Buy walls add positive score
        for wall in buy_walls:
            volume_ratio = wall.get('volume_ratio', 1)
            score += min(volume_ratio, 5)  # Cap individual wall contribution
        
        # Sell walls add negative score
        for wall in sell_walls:
            volume_ratio = wall.get('volume_ratio', 1)
            score -= min(volume_ratio, 5)
        
        return max(min(score, 10), -10)
    
    def _score_depth(self, metrics: Dict) -> float:
        """
        Score based on market depth (-10 to +10).
        
        More bid depth than ask depth: Bullish
        More ask depth than bid depth: Bearish
        """
        bid_depth = metrics.get('bid_depth_5pct', 0)
        ask_depth = metrics.get('ask_depth_5pct', 0)
        
        total_depth = bid_depth + ask_depth
        
        if total_depth <= 0:
            return 0.0
        
        # Calculate depth ratio
        depth_ratio = (bid_depth - ask_depth) / total_depth
        
        # Scale to -10 to +10
        return depth_ratio * 10
    
    def _determine_signal(
        self,
        overall_score: float,
        metrics: Dict
    ) -> Tuple[str, float]:
        """
        Determine trading signal and confidence.
        
        Args:
            overall_score: Overall orderbook score
            metrics: Calculated metrics
            
        Returns:
            Tuple of (signal, confidence)
        """
        # Determine signal
        if overall_score >= 2:
            signal = 'buy'
        elif overall_score <= -2:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Calculate confidence
        # Higher confidence with larger imbalance and presence of walls
        imbalance = abs(metrics.get('imbalance_ratio', 0))
        has_walls = metrics.get('has_buy_wall', False) or metrics.get('has_sell_wall', False)
        
        base_confidence = min(imbalance * 2, 0.5)
        wall_bonus = 0.2 if has_walls else 0
        magnitude_bonus = min(abs(overall_score) / 20, 0.3)
        
        confidence = base_confidence + wall_bonus + magnitude_bonus
        
        return signal, round(min(confidence, 1.0), 2)
    
    def get_analysis_summary(self, score: OrderbookScore) -> Dict:
        """
        Get human-readable analysis summary.
        
        Args:
            score: OrderbookScore object
            
        Returns:
            Dictionary with summary
        """
        signal_descriptions = {
            'buy': 'BULLISH - Buying pressure dominant',
            'sell': 'BEARISH - Selling pressure dominant',
            'neutral': 'NEUTRAL - Balanced order book',
        }
        
        interpretation = ""
        if score.overall_score >= 5:
            interpretation = "Strong buying pressure with significant support"
        elif score.overall_score >= 2:
            interpretation = "Moderate buying pressure"
        elif score.overall_score >= -2:
            interpretation = "Balanced market with no clear direction"
        elif score.overall_score >= -5:
            interpretation = "Moderate selling pressure"
        else:
            interpretation = "Strong selling pressure with significant resistance"
        
        return {
            'overall_score': score.overall_score,
            'signal': signal_descriptions.get(score.signal, score.signal),
            'confidence_percent': int(score.confidence * 100),
            'interpretation': interpretation,
            'metrics': {
                'total_bid_volume': score.metrics.get('total_bid_volume', 0),
                'total_ask_volume': score.metrics.get('total_ask_volume', 0),
                'imbalance_ratio': round(score.metrics.get('imbalance_ratio', 0), 4),
                'spread': score.metrics.get('spread', 0),
                'spread_percent': round(score.metrics.get('spread_percent', 0), 4),
                'buy_walls_count': len(score.metrics.get('buy_walls', [])),
                'sell_walls_count': len(score.metrics.get('sell_walls', [])),
            },
            'component_scores': score.details.get('component_scores', {}),
        }


def analyze_orderbook(ocr_text: str) -> Dict:
    """
    Convenience function to analyze orderbook.
    
    Args:
        ocr_text: OCR text from orderbook screenshot
        
    Returns:
        Dictionary with analysis results
    """
    engine = OrderbookEngine()
    score = engine.analyze_from_text(ocr_text)
    return engine.get_analysis_summary(score)
