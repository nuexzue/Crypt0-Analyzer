"""
LLM Engine
Integrates with local Ollama instance for AI-powered analysis
"""

import requests
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Container for LLM response."""
    success: bool
    report: str
    raw_response: Optional[Dict] = None
    error: Optional[str] = None


class LLMEngine:
    """
    LLM Engine for generating professional trading reports
    using local Ollama instance.
    """
    
    DEFAULT_MODEL = "mistral"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_URL = "http://localhost:11434/api/generate"
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        timeout: int = 120
    ):
        """
        Initialize LLM Engine.
        
        Args:
            base_url: Ollama API URL
            model: Model name to use
            temperature: Generation temperature
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or self.DEFAULT_URL
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.timeout = timeout
    
    def generate_report(
        self,
        technical_summary: Dict,
        orderbook_summary: Dict,
        final_score: Dict
    ) -> LLMResponse:
        """
        Generate a professional trading report using LLM.
        
        Args:
            technical_summary: Technical analysis summary
            orderbook_summary: Orderbook analysis summary
            final_score: Combined final score summary
            
        Returns:
            LLMResponse with generated report
        """
        prompt = self._build_report_prompt(
            technical_summary,
            orderbook_summary,
            final_score
        )
        
        return self._call_llm(prompt)
    
    def _build_report_prompt(
        self,
        technical_summary: Dict,
        orderbook_summary: Dict,
        final_score: Dict
    ) -> str:
        """
        Build the prompt for report generation.
        
        Args:
            technical_summary: Technical analysis data
            orderbook_summary: Orderbook analysis data
            final_score: Combined score data
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a professional cryptocurrency trading analyst. Generate a comprehensive trading report based on the following analysis data.

## TECHNICAL ANALYSIS DATA:
- Overall Technical Score: {technical_summary.get('overall_score', 'N/A')}/10
- Signal: {technical_summary.get('signal', 'N/A')}
- Confidence: {technical_summary.get('confidence_percent', 'N/A')}%
- Interpretation: {technical_summary.get('interpretation', 'N/A')}

Component Scores:
{json.dumps(technical_summary.get('component_scores', {}), indent=2)}

## ORDERBOOK ANALYSIS DATA:
- Overall Orderbook Score: {orderbook_summary.get('overall_score', 'N/A')}/10
- Signal: {orderbook_summary.get('signal', 'N/A')}
- Confidence: {orderbook_summary.get('confidence_percent', 'N/A')}%
- Interpretation: {orderbook_summary.get('interpretation', 'N/A')}

Orderbook Metrics:
- Total Bid Volume: {orderbook_summary.get('metrics', {}).get('total_bid_volume', 'N/A')}
- Total Ask Volume: {orderbook_summary.get('metrics', {}).get('total_ask_volume', 'N/A')}
- Imbalance Ratio: {orderbook_summary.get('metrics', {}).get('imbalance_ratio', 'N/A')}
- Spread: {orderbook_summary.get('metrics', {}).get('spread_percent', 'N/A')}%
- Buy Walls: {orderbook_summary.get('metrics', {}).get('buy_walls_count', 0)}
- Sell Walls: {orderbook_summary.get('metrics', {}).get('sell_walls_count', 0)}

## COMBINED SCORE:
- Final Score: {final_score.get('final_score', 'N/A')}/10
- Risk Grade: {final_score.get('risk_grade', 'N/A')}
- Overall Signal: {final_score.get('signal', 'N/A')}
- Confidence: {final_score.get('confidence', 'N/A')}
- Signals Aligned: {final_score.get('signals_aligned', 'N/A')}

---

Generate a professional trading report with the following sections:

1. **EXECUTIVE SUMMARY** (2-3 sentences)
2. **TECHNICAL ANALYSIS OVERVIEW** (Key findings from indicators)
3. **ORDERBOOK ANALYSIS OVERVIEW** (Market depth and pressure analysis)
4. **RISK ASSESSMENT** (Potential risks and concerns)
5. **TRADING RECOMMENDATION** (Specific actionable advice)
6. **KEY LEVELS TO WATCH** (If identifiable from data)

Use professional financial language. Be concise but thorough.
Format the response in clean markdown."""

        return prompt
    
    def _call_llm(self, prompt: str) -> LLMResponse:
        """
        Call the LLM API.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLMResponse object
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                report = result.get('response', '')
                
                return LLMResponse(
                    success=True,
                    report=report,
                    raw_response=result
                )
            else:
                return LLMResponse(
                    success=False,
                    report="",
                    error=f"API returned status {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                success=False,
                report="",
                error="Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
            )
        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                report="",
                error="Request timed out. The model may be loading or processing a large request."
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                report="",
                error=f"Unexpected error: {str(e)}"
            )
    
    def generate_quick_analysis(
        self,
        score: float,
        signal: str,
        risk_grade: str
    ) -> LLMResponse:
        """
        Generate a quick analysis summary.
        
        Args:
            score: Final score (0-10)
            signal: Trading signal
            risk_grade: Risk grade
            
        Returns:
            LLMResponse with quick analysis
        """
        prompt = f"""You are a crypto trading analyst. Provide a brief 2-3 sentence analysis:

Score: {score}/10
Signal: {signal}
Risk Grade: {risk_grade}

Give a concise professional interpretation suitable for a trading dashboard."""

        return self._call_llm(prompt)
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is accessible.
        
        Returns:
            True if connection successful
        """
        try:
            # Try to hit the Ollama API tags endpoint
            response = requests.get(
                self.base_url.replace('/api/generate', '/api/tags'),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_available_models(self) -> list:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                self.base_url.replace('/api/generate', '/api/tags'),
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []


def generate_ai_report(
    technical_summary: Dict,
    orderbook_summary: Dict,
    final_score: Dict,
    model: str = "mistral",
    temperature: float = 0.3
) -> Dict:
    """
    Convenience function to generate AI report.
    
    Args:
        technical_summary: Technical analysis summary
        orderbook_summary: Orderbook analysis summary
        final_score: Final score summary
        model: Model to use
        temperature: Generation temperature
        
    Returns:
        Dictionary with report and status
    """
    engine = LLMEngine(model=model, temperature=temperature)
    response = engine.generate_report(technical_summary, orderbook_summary, final_score)
    
    return {
        'success': response.success,
        'report': response.report,
        'error': response.error,
    }
