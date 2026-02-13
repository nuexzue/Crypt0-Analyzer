"""
Scoring Engine
Combines technical and orderbook scores into final trading score
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .technical_engine import TechnicalScore, TechnicalEngine
from .orderbook_engine import OrderbookScore, OrderbookEngine


class RiskGrade(Enum):
    """Risk grade enumeration."""
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


@dataclass
class FinalScore:
    """Container for final combined score."""
    final_score: float  # 0 to 10
    technical_score: float  # -10 to +10 (original)
    orderbook_score: float  # -10 to +10 (original)
    risk_grade: str
    signal: str
    confidence: float
    recommendation: str
    details: Dict


class ScoringEngine:
    """
    Scoring engine that combines technical and orderbook analysis
    into a final normalized score with risk grading.
    """
    
    # Weights for combining scores
    TECHNICAL_WEIGHT = 0.6
    ORDERBOOK_WEIGHT = 0.4
    
    def __init__(
        self,
        technical_weight: float = 0.6,
        orderbook_weight: float = 0.4
    ):
        """
        Initialize scoring engine.
        
        Args:
            technical_weight: Weight for technical score (default 0.6)
            orderbook_weight: Weight for orderbook score (default 0.4)
        """
        if not 0.99 <= technical_weight + orderbook_weight <= 1.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.technical_weight = technical_weight
        self.orderbook_weight = orderbook_weight
        self.technical_engine = TechnicalEngine()
        self.orderbook_engine = OrderbookEngine()
    
    def calculate_final_score(
        self,
        technical_score: TechnicalScore,
        orderbook_score: OrderbookScore
    ) -> FinalScore:
        """
        Calculate final combined score.
        
        Args:
            technical_score: Technical analysis score
            orderbook_score: Orderbook analysis score
            
        Returns:
            FinalScore object
        """
        # Get raw scores (-10 to +10)
        tech_raw = technical_score.overall_score
        ob_raw = orderbook_score.overall_score
        
        # Calculate weighted combined score (-10 to +10)
        combined_raw = (
            tech_raw * self.technical_weight +
            ob_raw * self.orderbook_weight
        )
        
        # Normalize to 0-10 scale
        final_score = self._normalize_score(combined_raw)
        
        # Calculate risk grade
        risk_grade = self._calculate_risk_grade(final_score, technical_score, orderbook_score)
        
        # Determine signal from combined score
        signal = self._determine_signal(combined_raw)
        
        # Calculate combined confidence
        confidence = self._combine_confidence(
            technical_score.confidence,
            orderbook_score.confidence
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            final_score, signal, risk_grade, confidence
        )
        
        # Build details
        details = {
            'raw_combined_score': round(combined_raw, 2),
            'technical_contribution': round(tech_raw * self.technical_weight, 2),
            'orderbook_contribution': round(ob_raw * self.orderbook_weight, 2),
            'technical_signal': technical_score.signal,
            'orderbook_signal': orderbook_score.signal,
            'signals_aligned': technical_score.signal == orderbook_score.signal,
        }
        
        return FinalScore(
            final_score=round(final_score, 2),
            technical_score=tech_raw,
            orderbook_score=ob_raw,
            risk_grade=risk_grade.value,
            signal=signal,
            confidence=confidence,
            recommendation=recommendation,
            details=details
        )
    
    def calculate_from_text(
        self,
        chart_text: str,
        orderbook_text: str
    ) -> FinalScore:
        """
        Calculate final score from OCR text.
        
        Args:
            chart_text: OCR text from chart screenshot
            orderbook_text: OCR text from orderbook screenshot
            
        Returns:
            FinalScore object
        """
        technical_score = self.technical_engine.analyze_from_text(chart_text)
        orderbook_score = self.orderbook_engine.analyze_from_text(orderbook_text)
        
        return self.calculate_final_score(technical_score, orderbook_score)
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize score from -10/+10 range to 0-10 range.
        
        Args:
            raw_score: Score in -10 to +10 range
            
        Returns:
            Score in 0 to 10 range
        """
        # Map -10 to 0, 0 to 5, +10 to 10
        normalized = (raw_score + 10) / 2
        return max(0, min(10, normalized))
    
    def _calculate_risk_grade(
        self,
        final_score: float,
        technical_score: TechnicalScore,
        orderbook_score: OrderbookScore
    ) -> RiskGrade:
        """
        Calculate risk grade based on scores and signal alignment.
        
        Args:
            final_score: Normalized final score
            technical_score: Technical analysis results
            orderbook_score: Orderbook analysis results
            
        Returns:
            RiskGrade enum value
        """
        # Check signal alignment
        signals_aligned = technical_score.signal == orderbook_score.signal
        
        # Average confidence
        avg_confidence = (technical_score.confidence + orderbook_score.confidence) / 2
        
        # Base grade from score
        if final_score >= 8.5:
            base_grade = RiskGrade.A_PLUS
        elif final_score >= 7.5:
            base_grade = RiskGrade.A
        elif final_score >= 6.5:
            base_grade = RiskGrade.B_PLUS
        elif final_score >= 5.5:
            base_grade = RiskGrade.B
        elif final_score >= 4.5:
            base_grade = RiskGrade.C
        elif final_score >= 3.0:
            base_grade = RiskGrade.D
        else:
            base_grade = RiskGrade.F
        
        # Adjust for misaligned signals
        if not signals_aligned and avg_confidence > 0.5:
            # Downgrade by one level for conflicting signals
            grade_order = list(RiskGrade)
            current_idx = grade_order.index(base_grade)
            if current_idx < len(grade_order) - 1:
                base_grade = grade_order[current_idx + 1]
        
        return base_grade
    
    def _determine_signal(self, combined_score: float) -> str:
        """
        Determine trading signal from combined score.
        
        Args:
            combined_score: Raw combined score (-10 to +10)
            
        Returns:
            Signal string
        """
        if combined_score >= 3:
            return 'strong_buy'
        elif combined_score >= 1:
            return 'buy'
        elif combined_score >= -1:
            return 'neutral'
        elif combined_score >= -3:
            return 'sell'
        else:
            return 'strong_sell'
    
    def _combine_confidence(
        self,
        tech_confidence: float,
        ob_confidence: float
    ) -> float:
        """
        Combine confidence scores.
        
        Args:
            tech_confidence: Technical analysis confidence
            ob_confidence: Orderbook analysis confidence
            
        Returns:
            Combined confidence
        """
        # Weighted average with slight bonus for agreement
        weighted = (
            tech_confidence * self.technical_weight +
            ob_confidence * self.orderbook_weight
        )
        
        return round(min(weighted, 1.0), 2)
    
    def _generate_recommendation(
        self,
        final_score: float,
        signal: str,
        risk_grade: RiskGrade,
        confidence: float
    ) -> str:
        """
        Generate human-readable recommendation.
        
        Args:
            final_score: Final normalized score
            signal: Trading signal
            risk_grade: Risk grade
            confidence: Confidence level
            
        Returns:
            Recommendation string
        """
        signal_actions = {
            'strong_buy': "STRONG BUY - Consider entering a long position",
            'buy': "BUY - Favorable conditions for long entry",
            'neutral': "HOLD - Wait for clearer signals",
            'sell': "SELL - Consider reducing position",
            'strong_sell': "STRONG SELL - Consider exiting or shorting",
        }
        
        action = signal_actions.get(signal, "HOLD")
        
        confidence_text = ""
        if confidence >= 0.7:
            confidence_text = "High confidence in analysis."
        elif confidence >= 0.4:
            confidence_text = "Moderate confidence - use additional confirmation."
        else:
            confidence_text = "Low confidence - exercise caution."
        
        risk_text = f"Risk Grade: {risk_grade.value}"
        
        return f"{action}\n{confidence_text}\n{risk_text}"
    
    def get_score_summary(self, score: FinalScore) -> Dict:
        """
        Get comprehensive score summary.
        
        Args:
            score: FinalScore object
            
        Returns:
            Dictionary with summary
        """
        signal_labels = {
            'strong_buy': 'STRONG BUY 📈',
            'buy': 'BUY ↑',
            'neutral': 'NEUTRAL ↔',
            'sell': 'SELL ↓',
            'strong_sell': 'STRONG SELL 📉',
        }
        
        return {
            'final_score': score.final_score,
            'final_score_display': f"{score.final_score}/10",
            'technical_score': score.technical_score,
            'orderbook_score': score.orderbook_score,
            'risk_grade': score.risk_grade,
            'signal': signal_labels.get(score.signal, score.signal),
            'confidence': f"{int(score.confidence * 100)}%",
            'recommendation': score.recommendation,
            'technical_weight': f"{int(self.technical_weight * 100)}%",
            'orderbook_weight': f"{int(self.orderbook_weight * 100)}%",
            'signals_aligned': score.details.get('signals_aligned', False),
            'raw_combined': score.details.get('raw_combined_score', 0),
        }


def calculate_final_score(
    chart_text: str,
    orderbook_text: str
) -> Dict:
    """
    Convenience function to calculate final score.
    
    Args:
        chart_text: OCR text from chart
        orderbook_text: OCR text from orderbook
        
    Returns:
        Dictionary with score summary
    """
    engine = ScoringEngine()
    score = engine.calculate_from_text(chart_text, orderbook_text)
    return engine.get_score_summary(score)
