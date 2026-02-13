"""
Technical Analysis Engine
Analyzes technical indicators and produces weighted scores
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .indicator_parser import IndicatorData, IndicatorParser


@dataclass
class TechnicalScore:
    """Container for technical analysis scores."""
    overall_score: float  # -10 to +10
    rsi_score: float
    macd_score: float
    ema_score: float
    bollinger_score: float
    volume_score: float
    signal: str  # 'buy', 'sell', 'neutral'
    confidence: float  # 0 to 1
    details: Dict


class TechnicalEngine:
    """
    Technical analysis engine that computes weighted scores
    from various technical indicators.
    """
    
    # Default weights for different indicators
    DEFAULT_WEIGHTS = {
        'rsi': 0.25,
        'macd': 0.25,
        'ema': 0.20,
        'bollinger': 0.15,
        'volume': 0.15,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize technical engine.
        
        Args:
            weights: Custom weights for indicators (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()
        self.parser = IndicatorParser()
    
    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def analyze(self, indicator_data: IndicatorData) -> TechnicalScore:
        """
        Perform technical analysis on indicator data.
        
        Args:
            indicator_data: Parsed indicator data
            
        Returns:
            TechnicalScore with overall and component scores
        """
        # Calculate individual scores
        rsi_score = self._score_rsi(indicator_data.rsi)
        macd_score = self._score_macd(
            indicator_data.macd_value,
            indicator_data.macd_signal,
            indicator_data.macd_histogram
        )
        ema_score = self._score_ema(
            indicator_data.price,
            indicator_data.ema_12,
            indicator_data.ema_26,
            indicator_data.ema_50,
            indicator_data.ema_200
        )
        bollinger_score = self._score_bollinger(
            indicator_data.price,
            indicator_data.bollinger_upper,
            indicator_data.bollinger_middle,
            indicator_data.bollinger_lower
        )
        volume_score = self._score_volume(
            indicator_data.volume,
            indicator_data.volume_ma
        )
        
        # Calculate weighted overall score
        scores = {
            'rsi': rsi_score,
            'macd': macd_score,
            'ema': ema_score,
            'bollinger': bollinger_score,
            'volume': volume_score,
        }
        
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        
        if valid_scores:
            # Renormalize weights for available scores
            total_weight = sum(self.weights[k] for k in valid_scores)
            overall_score = sum(
                v * (self.weights[k] / total_weight)
                for k, v in valid_scores.items()
            )
        else:
            overall_score = 0.0
        
        # Determine signal and confidence
        signal, confidence = self._determine_signal(overall_score, valid_scores)
        
        # Build details
        details = {
            'component_scores': scores,
            'indicator_values': {
                'rsi': indicator_data.rsi,
                'macd': indicator_data.macd_value,
                'macd_signal': indicator_data.macd_signal,
                'macd_histogram': indicator_data.macd_histogram,
                'price': indicator_data.price,
                'ema_12': indicator_data.ema_12,
                'ema_26': indicator_data.ema_26,
                'ema_50': indicator_data.ema_50,
                'ema_200': indicator_data.ema_200,
                'bb_upper': indicator_data.bollinger_upper,
                'bb_middle': indicator_data.bollinger_middle,
                'bb_lower': indicator_data.bollinger_lower,
                'volume': indicator_data.volume,
                'volume_ma': indicator_data.volume_ma,
            },
            'valid_indicators': len(valid_scores),
            'total_indicators': len(scores),
        }
        
        return TechnicalScore(
            overall_score=round(overall_score, 2),
            rsi_score=rsi_score if rsi_score is not None else 0,
            macd_score=macd_score if macd_score is not None else 0,
            ema_score=ema_score if ema_score is not None else 0,
            bollinger_score=bollinger_score if bollinger_score is not None else 0,
            volume_score=volume_score if volume_score is not None else 0,
            signal=signal,
            confidence=confidence,
            details=details
        )
    
    def analyze_from_text(self, ocr_text: str) -> TechnicalScore:
        """
        Analyze technical indicators from OCR text.
        
        Args:
            ocr_text: Raw OCR text from chart screenshot
            
        Returns:
            TechnicalScore
        """
        indicator_data = self.parser.parse(ocr_text)
        return self.analyze(indicator_data)
    
    def _score_rsi(self, rsi: Optional[float]) -> Optional[float]:
        """
        Score RSI indicator (-10 to +10).
        
        RSI < 30: Oversold (bullish) -> positive score
        RSI > 70: Overbought (bearish) -> negative score
        RSI 30-70: Neutral zone
        """
        if rsi is None:
            return None
        
        if rsi < 20:
            return 10.0  # Extremely oversold - strong buy
        elif rsi < 30:
            return 7.0 + (30 - rsi) / 10 * 3  # Oversold
        elif rsi < 40:
            return 3.0 + (40 - rsi) / 10 * 4  # Slightly oversold
        elif rsi <= 60:
            return (50 - rsi) / 10 * 3  # Neutral zone
        elif rsi < 70:
            return -3.0 - (rsi - 60) / 10 * 4  # Slightly overbought
        elif rsi < 80:
            return -7.0 - (rsi - 70) / 10 * 3  # Overbought
        else:
            return -10.0  # Extremely overbought - strong sell
    
    def _score_macd(
        self,
        macd_value: Optional[float],
        signal: Optional[float],
        histogram: Optional[float]
    ) -> Optional[float]:
        """
        Score MACD indicator (-10 to +10).
        
        MACD > Signal + positive histogram: Bullish
        MACD < Signal + negative histogram: Bearish
        """
        if macd_value is None:
            return None
        
        score = 0.0
        
        # MACD line position
        if macd_value > 0:
            score += min(macd_value * 2, 3)  # Bullish
        else:
            score += max(macd_value * 2, -3)  # Bearish
        
        # MACD vs Signal crossover
        if signal is not None:
            diff = macd_value - signal
            if diff > 0:
                score += min(diff * 5, 4)  # Bullish crossover
            else:
                score += max(diff * 5, -4)  # Bearish crossover
        
        # Histogram momentum
        if histogram is not None:
            if histogram > 0:
                score += min(histogram * 3, 3)  # Bullish momentum
            else:
                score += max(histogram * 3, -3)  # Bearish momentum
        
        return max(min(score, 10), -10)
    
    def _score_ema(
        self,
        price: Optional[float],
        ema_12: Optional[float],
        ema_26: Optional[float],
        ema_50: Optional[float],
        ema_200: Optional[float]
    ) -> Optional[float]:
        """
        Score EMA indicators (-10 to +10).
        
        Price above EMAs: Bullish
        Price below EMAs: Bearish
        EMAs in bullish order (12 > 26 > 50 > 200): Strong bullish
        """
        available_emas = [e for e in [ema_12, ema_26, ema_50, ema_200] if e is not None]
        
        if not available_emas or price is None:
            return None
        
        score = 0.0
        
        # Price position relative to EMAs
        above_count = sum(1 for ema in available_emas if price > ema)
        below_count = len(available_emas) - above_count
        
        score += (above_count - below_count) * 2
        
        # EMA alignment
        if ema_12 is not None and ema_26 is not None:
            if ema_12 > ema_26:
                score += 2  # Short-term bullish
            else:
                score -= 2  # Short-term bearish
        
        if ema_50 is not None and ema_200 is not None:
            if ema_50 > ema_200:
                score += 2  # Long-term bullish (golden cross potential)
            else:
                score -= 2  # Long-term bearish (death cross potential)
        
        return max(min(score, 10), -10)
    
    def _score_bollinger(
        self,
        price: Optional[float],
        upper: Optional[float],
        middle: Optional[float],
        lower: Optional[float]
    ) -> Optional[float]:
        """
        Score Bollinger Bands (-10 to +10).
        
        Price near/below lower band: Oversold (bullish)
        Price near/above upper band: Overbought (bearish)
        """
        if price is None or (upper is None and middle is None and lower is None):
            return None
        
        score = 0.0
        
        if upper is not None and lower is not None:
            band_width = upper - lower
            
            if band_width > 0:
                # Position within bands (0 = lower, 1 = upper)
                if lower <= price <= upper:
                    position = (price - lower) / band_width
                elif price < lower:
                    position = 0
                else:
                    position = 1
                
                # Score based on position
                # Near lower (oversold) is bullish, near upper (overbought) is bearish
                score = (0.5 - position) * 20  # -10 to +10
        
        elif middle is not None:
            # Only middle band available
            diff_percent = (price - middle) / middle * 100
            score = -diff_percent * 2  # Below middle is bullish
        
        return max(min(score, 10), -10)
    
    def _score_volume(
        self,
        volume: Optional[float],
        volume_ma: Optional[float]
    ) -> Optional[float]:
        """
        Score volume indicator (-10 to +10).
        
        High volume relative to MA: Confirms trend
        Low volume: Weak signal
        """
        if volume is None:
            return None
        
        if volume_ma is None:
            # Without MA, just return neutral
            return 0.0
        
        if volume_ma <= 0:
            return 0.0
        
        ratio = volume / volume_ma
        
        if ratio > 2.0:
            # Very high volume - strong confirmation
            return 8.0
        elif ratio > 1.5:
            # High volume
            return 5.0
        elif ratio > 1.0:
            # Above average
            return 2.0
        elif ratio > 0.5:
            # Below average
            return -2.0
        else:
            # Very low volume - weak signal
            return -5.0
    
    def _determine_signal(
        self,
        overall_score: float,
        valid_scores: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Determine trading signal and confidence.
        
        Args:
            overall_score: Weighted overall score
            valid_scores: Dictionary of valid component scores
            
        Returns:
            Tuple of (signal, confidence)
        """
        # Determine signal based on score
        if overall_score >= 3:
            signal = 'buy'
        elif overall_score <= -3:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        # Calculate confidence based on score consistency and magnitude
        if not valid_scores:
            return signal, 0.0
        
        # Base confidence on number of agreeing indicators
        positive_count = sum(1 for v in valid_scores.values() if v > 0)
        negative_count = sum(1 for v in valid_scores.values() if v < 0)
        total_count = len(valid_scores)
        
        agreement_ratio = max(positive_count, negative_count) / total_count
        
        # Scale confidence by score magnitude
        magnitude_factor = min(abs(overall_score) / 10, 1.0)
        
        confidence = agreement_ratio * 0.6 + magnitude_factor * 0.4
        
        return signal, round(confidence, 2)
    
    def get_analysis_summary(self, score: TechnicalScore) -> Dict:
        """
        Get a human-readable analysis summary.
        
        Args:
            score: TechnicalScore object
            
        Returns:
            Dictionary with summary information
        """
        signal_descriptions = {
            'buy': 'BULLISH - Consider buying',
            'sell': 'BEARISH - Consider selling',
            'neutral': 'NEUTRAL - No clear direction',
        }
        
        score_interpretation = ""
        if score.overall_score >= 7:
            score_interpretation = "Strong bullish signals across indicators"
        elif score.overall_score >= 3:
            score_interpretation = "Moderate bullish bias"
        elif score.overall_score >= -3:
            score_interpretation = "Mixed or neutral signals"
        elif score.overall_score >= -7:
            score_interpretation = "Moderate bearish bias"
        else:
            score_interpretation = "Strong bearish signals across indicators"
        
        return {
            'overall_score': score.overall_score,
            'signal': signal_descriptions.get(score.signal, score.signal),
            'confidence_percent': int(score.confidence * 100),
            'interpretation': score_interpretation,
            'component_scores': {
                'RSI': score.rsi_score,
                'MACD': score.macd_score,
                'EMA': score.ema_score,
                'Bollinger': score.bollinger_score,
                'Volume': score.volume_score,
            },
            'valid_indicators': score.details.get('valid_indicators', 0),
        }


def analyze_technical(ocr_text: str) -> Dict:
    """
    Convenience function to analyze technical indicators.
    
    Args:
        ocr_text: OCR text from chart screenshot
        
    Returns:
        Dictionary with analysis results
    """
    engine = TechnicalEngine()
    score = engine.analyze_from_text(ocr_text)
    return engine.get_analysis_summary(score)
