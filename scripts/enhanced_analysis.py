"""
Enhanced Analysis Module

Combines audio quality metrics with transcript analysis to compute a comprehensive
confidence score and provides AI-powered buy/sell recommendations.

Features:
- Detailed transcript analysis (numbers, financial details, descriptions)
- SNR-based audio quality scoring
- Combined audio + transcript confidence score
- AI-powered analysis and recommendations
"""

import re
import logging
from typing import Dict, List, Any
from mistralai import Mistral
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MISTRAL_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_financial_numbers(text: str) -> Dict[str, Any]:
    """
    Extract and analyze financial numbers from transcript.
    
    Args:
        text: Transcript text
        
    Returns:
        Dictionary with extracted numbers and metrics
    """
    # Extract dollar amounts
    dollar_pattern = r'\$[\d,]+\.?\d*|\$[\d,]+'
    dollar_matches = re.findall(dollar_pattern, text)
    
    # Extract percentages
    percent_pattern = r'\d+\.?\d*%'
    percent_matches = re.findall(percent_pattern, text)
    
    # Extract raw numbers (potential metrics)
    number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    number_matches = re.findall(number_pattern, text)
    
    # Extract financial keywords
    financial_keywords = [
        'revenue', 'earnings', 'profit', 'margin', 'growth', 'income',
        'cash', 'debt', 'assets', 'liabilities', 'equity', 'capital',
        'ebitda', 'operating', 'net', 'gross', 'quarter', 'year', 'billion',
        'million', 'thousand', 'increase', 'decrease', 'rise', 'fall', 'beat',
        'miss', 'guidance', 'outlook', 'forecast', 'expect', 'project'
    ]
    
    keyword_counts = {}
    for keyword in financial_keywords:
        pattern = r'\b' + keyword + r'\b'
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count > 0:
            keyword_counts[keyword] = count
    
    return {
        'dollar_amounts': dollar_matches,
        'dollar_count': len(dollar_matches),
        'percentages': percent_matches,
        'percentage_count': len(percent_matches),
        'numbers': number_matches,
        'number_count': len(number_matches),
        'financial_keywords': keyword_counts,
        'total_financial_keywords': sum(keyword_counts.values())
    }


def analyze_transcript_detail(text: str) -> Dict[str, Any]:
    """
    Analyze the level of detail in the transcript.
    
    Args:
        text: Transcript text
        
    Returns:
        Dictionary with detail metrics
    """
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Word count
    words = text.split()
    
    # Average sentence length
    avg_sentence_length = len(words) / max(len(sentences), 1)
    
    # Quantitative detail score (based on numbers and specific metrics)
    financial_data = extract_financial_numbers(text)
    
    # Detail score components
    number_density = financial_data['number_count'] / max(len(words), 1) * 100
    keyword_density = financial_data['total_financial_keywords'] / max(len(words), 1) * 100
    
    # Specificity score (0-1)
    specificity_score = min(1.0, (
        (financial_data['dollar_count'] * 0.3) +
        (financial_data['percentage_count'] * 0.3) +
        (financial_data['total_financial_keywords'] * 0.4)
    ) / max(1, len(sentences)))
    
    return {
        'sentence_count': len(sentences),
        'word_count': len(words),
        'avg_sentence_length': avg_sentence_length,
        'number_density': number_density,
        'keyword_density': keyword_density,
        'specificity_score': specificity_score,
        'financial_data': financial_data
    }


def compute_combined_score(
    audio_metrics: Dict[str, Any],
    transcript_metrics: Dict[str, Any],
    nlp_signals: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute a combined confidence score from audio and transcript analysis.
    
    Args:
        audio_metrics: Audio quality metrics (SNR, pauses, etc.)
        transcript_metrics: Transcript detail analysis
        nlp_signals: NLP sentiment and confidence signals
        
    Returns:
        Combined score and analysis
    """
    # Extract audio quality score
    snr_db = audio_metrics.get('snr', {}).get('snr_db', 0)
    voice_ratio = audio_metrics.get('vad', {}).get('voice_activity_ratio', 0.5)
    pauses_over_500 = audio_metrics.get('pauses', {}).get('pauses_over_500ms', 0)
    pauses_over_1000 = audio_metrics.get('pauses', {}).get('pauses_over_1000ms', 0)
    
    # Normalize SNR to 0-1 (assuming 30dB is excellent)
    snr_score = min(1.0, max(0.0, snr_db / 30.0))
    
    # Hesitation penalty from pauses
    hesitation_penalty = (pauses_over_500 * 0.05) + (pauses_over_1000 * 0.1)
    hesitation_penalty = min(0.5, hesitation_penalty)
    
    # Audio confidence score (0-1)
    audio_confidence = (snr_score * 0.6) + (voice_ratio * 0.4) - hesitation_penalty
    audio_confidence = max(0.0, min(1.0, audio_confidence))
    
    # Transcript specificity score
    specificity = transcript_metrics.get('specificity_score', 0.5)
    
    # NLP confidence (normalized to 0-1)
    nlp_conf = nlp_signals.get('confidence_score', 0.0)
    nlp_conf_normalized = (nlp_conf + 1) / 2  # Convert -1 to 1 range to 0 to 1
    
    # Sentiment from NLP
    sentiment = nlp_signals.get('remarks_sentiment', 0.0)
    sentiment_normalized = (sentiment + 1) / 2
    
    # Combined score weights
    # - Audio quality: 25% (is the speaker confident in delivery?)
    # - Transcript specificity: 25% (are they providing detailed numbers?)
    # - NLP confidence: 25% (linguistic confidence indicators)
    # - Sentiment: 25% (is the content positive/negative?)
    
    combined_score = (
        audio_confidence * 0.25 +
        specificity * 0.25 +
        nlp_conf_normalized * 0.25 +
        sentiment_normalized * 0.25
    )
    
    # Scale back to -1 to 1 range for consistency
    combined_score_scaled = (combined_score * 2) - 1.0
    
    return {
        'combined_confidence': combined_score_scaled,
        'audio_confidence': audio_confidence,
        'transcript_specificity': specificity,
        'nlp_confidence': nlp_conf_normalized,
        'sentiment_score': sentiment_normalized,
        'snr_db': snr_db,
        'hesitation_penalty': hesitation_penalty,
        'breakdown': {
            'audio_weight': 0.25,
            'specificity_weight': 0.25,
            'nlp_weight': 0.25,
            'sentiment_weight': 0.25
        }
    }


def generate_ai_analysis(
    combined_score: float,
    audio_metrics: Dict[str, Any],
    transcript_metrics: Dict[str, Any],
    nlp_signals: Dict[str, Any],
    transcript_text: str
) -> str:
    """
    Generate AI-powered analysis and buy/sell recommendation using Mistral AI.
    
    Args:
        combined_score: Combined confidence score (-1 to 1)
        audio_metrics: Audio quality metrics
        transcript_metrics: Transcript detail analysis
        nlp_signals: NLP signals
        transcript_text: Original transcript text
        
    Returns:
        AI analysis paragraph with recommendation
    """
    if not MISTRAL_API_KEY:
        return "AI analysis not available. Please set MISTRAL_API_KEY in .env file."
    
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Prepare summary for the AI
        summary = f"""
Earnings Call Analysis Summary:

Combined Confidence Score: {combined_score:.2f} (-1 to 1 scale, where 1 is most confident)

Audio Quality Metrics:
- SNR: {audio_metrics.get('snr', {}).get('snr_db', 0):.2f} dB
- Voice Activity Ratio: {audio_metrics.get('vad', {}).get('voice_activity_ratio', 0):.2f}
- Hesitation Pauses (>500ms): {audio_metrics.get('pauses', {}).get('pauses_over_500ms', 0)}
- Hesitation Pauses (>1000ms): {audio_metrics.get('pauses', {}).get('pauses_over_1000ms', 0)}

Transcript Detail Metrics:
- Specificity Score: {transcript_metrics.get('specificity_score', 0):.2f}
- Financial Numbers Mentioned: {transcript_metrics.get('financial_data', {}).get('number_count', 0)}
- Dollar Amounts: {transcript_metrics.get('financial_data', {}).get('dollar_count', 0)}
- Percentages: {transcript_metrics.get('financial_data', {}).get('percentage_count', 0)}
- Financial Keywords: {transcript_metrics.get('financial_data', {}).get('total_financial_keywords', 0)}

NLP Signals:
- Remarks Sentiment: {nlp_signals.get('remarks_sentiment', 0):.2f}
- Q&A Sentiment: {nlp_signals.get('qa_sentiment', 0):.2f}
- Hedge Density: {nlp_signals.get('hedge_density', 0):.2f}
- CEO/CFO Divergence: {nlp_signals.get('ceo_cfo_divergence', 0):.2f}

Transcript Excerpt (first 500 chars):
{transcript_text[:500]}...
"""
        
        response = client.chat.complete(
            model="mistral-medium",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in earnings call analysis. Based on the provided metrics, provide a concise analysis (2-3 sentences) of what the confidence score means and whether to BUY, HOLD, or SELL the stock. Be direct and evidence-based."
                },
                {
                    "role": "user",
                    "content": summary
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return f"AI analysis unavailable: {str(e)}"
