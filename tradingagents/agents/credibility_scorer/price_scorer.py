"""
Price Credibility Scorer

This module identifies cross-report consensus on price-related signals from multiple
analyst reports. It detects agreement, discounts outliers, and produces a robust
consensus-based price signal for trading decisions.

Focus: Pure price-derived signals only (no fundamentals/news/sentiment)
"""

import re
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

from .utils import extract_report_to_dict, extract_decision_from_report, get_model_name


@dataclass
class PriceClaim:
    """Represents a normalized price-related claim from a report."""
    horizon: str  # '1_week', '1_month', '3_months', '1_year', '5_years', 'next_day'
    direction: str  # 'up', 'down', 'flat', 'uncertain'
    magnitude_min: float  # Minimum % change
    magnitude_max: float  # Maximum % change
    confidence: float  # 0-1 claim confidence from source
    source_report_id: int  # Which report this came from
    raw_text: str  # Original claim text
    claim_type: str  # 'trend', 'return', 'prediction', 'volatility'

    def magnitude_center(self) -> float:
        """Return the center of the magnitude range."""
        return (self.magnitude_min + self.magnitude_max) / 2

    def magnitude_range_width(self) -> float:
        """Return the width of the magnitude range."""
        return self.magnitude_max - self.magnitude_min


class PriceClaimExtractor:
    """Extracts and normalizes price claims from analyst reports."""

    # Horizon keywords mapping
    HORIZON_PATTERNS = {
        'next_day': [r'next\s*day', r'tomorrow', r'next\s*trading\s*day', r'1\s*day'],
        '1_week': [r'1\s*week', r'one\s*week', r'weekly', r'1w', r'7\s*day'],
        '1_month': [r'1\s*month', r'one\s*month', r'monthly', r'1m', r'30\s*day'],
        '3_months': [r'3\s*month', r'three\s*month', r'quarterly', r'3m', r'90\s*day', r'quarter'],
        '6_months': [r'6\s*month', r'six\s*month', r'half\s*year', r'6m', r'180\s*day', r'half\s*year'],
        '1_year': [r'1\s*year', r'one\s*year', r'yearly', r'annual', r'1y', r'12\s*month'],
    }

    # Direction keywords
    DIRECTION_UP = [r'\bup\b', r'bull(?:ish)?', r'positive', r'gain', r'increas', r'rise', r'uptrend', r'upward']
    DIRECTION_DOWN = [r'\bdown\b', r'bear(?:ish)?', r'negative', r'loss', r'decreas', r'fall', r'downtrend', r'downward', r'decline']
    DIRECTION_FLAT = [r'flat', r'sideways', r'neutral', r'stable', r'unchanged']

    @staticmethod
    def extract_claims_from_report(report_text: str, report_id: int) -> List[PriceClaim]:
        """
        Extract all price claims from a single report.

        Args:
            report_text: The report content
            report_id: Identifier for this report (0-indexed)

        Returns:
            List of normalized PriceClaim objects
        """
        claims = []

        # Split into sentences for processing
        sentences = re.split(r'[.!?\n]+', report_text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Skip if sentence is too short or doesn't mention price/trend/return
            if len(sentence) < 20:
                continue

            price_related = any(kw in sentence_lower for kw in [
                'price', 'return', 'trend', 'predict', 'direction', 'movement',
                'volatility', '%', 'percent', 'up', 'down', 'bull', 'bear'
            ])

            if not price_related:
                continue

            # Try to extract structured information
            horizon = PriceClaimExtractor._extract_horizon(sentence_lower)
            direction = PriceClaimExtractor._extract_direction(sentence_lower)
            magnitude = PriceClaimExtractor._extract_magnitude(sentence)
            confidence = PriceClaimExtractor._extract_confidence(sentence_lower)
            claim_type = PriceClaimExtractor._classify_claim_type(sentence_lower)

            # Only create claim if we extracted meaningful information
            if horizon and direction != 'uncertain':
                claims.append(PriceClaim(
                    horizon=horizon,
                    direction=direction,
                    magnitude_min=magnitude[0],
                    magnitude_max=magnitude[1],
                    confidence=confidence,
                    source_report_id=report_id,
                    raw_text=sentence.strip(),
                    claim_type=claim_type
                ))

        # Also try to extract from structured sections (tables, bullet points)
        claims.extend(PriceClaimExtractor._extract_from_structured_sections(report_text, report_id))

        return claims

    @staticmethod
    def _extract_horizon(text: str) -> Optional[str]:
        """Extract time horizon from text."""
        for horizon, patterns in PriceClaimExtractor.HORIZON_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return horizon
        return None

    @staticmethod
    def _extract_direction(text: str) -> str:
        """Extract directional signal from text."""
        # Check for up signals
        up_count = sum(1 for pattern in PriceClaimExtractor.DIRECTION_UP
                      if re.search(pattern, text, re.IGNORECASE))

        # Check for down signals
        down_count = sum(1 for pattern in PriceClaimExtractor.DIRECTION_DOWN
                        if re.search(pattern, text, re.IGNORECASE))

        # Check for flat signals
        flat_count = sum(1 for pattern in PriceClaimExtractor.DIRECTION_FLAT
                        if re.search(pattern, text, re.IGNORECASE))

        # Determine dominant direction
        if flat_count > 0 and flat_count >= up_count and flat_count >= down_count:
            return 'flat'
        elif up_count > down_count:
            return 'up'
        elif down_count > up_count:
            return 'down'
        else:
            return 'uncertain'

    @staticmethod
    def _extract_magnitude(text: str) -> Tuple[float, float]:
        """
        Extract magnitude range from text.
        Returns (min, max) tuple.
        """
        # Look for percentage patterns
        # Pattern 1: "X%" or "X percent"
        simple_pct = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        if simple_pct:
            values = [float(x) for x in simple_pct]
            if len(values) == 1:
                # Single value - assume small range around it
                return (values[0] * 0.8, values[0] * 1.2)
            else:
                # Multiple values - use min/max
                return (min(values), max(values))

        # Pattern 2: "between X% and Y%"
        range_match = re.search(r'between\s+(\d+(?:\.\d+)?)\s*%?\s+and\s+(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))

        # Pattern 3: "X-Y%"
        dash_range = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%', text)
        if dash_range:
            return (float(dash_range.group(1)), float(dash_range.group(2)))

        # Default: no specific magnitude mentioned
        return (0.0, 0.0)

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Extract confidence level from text (0-1 scale)."""
        # High confidence keywords
        if any(kw in text for kw in ['strong', 'high confidence', 'certain', 'definite', 'clear']):
            return 0.9

        # Medium confidence keywords
        if any(kw in text for kw in ['moderate', 'medium confidence', 'likely', 'probable']):
            return 0.6

        # Low confidence keywords
        if any(kw in text for kw in ['weak', 'low confidence', 'uncertain', 'unclear', 'mixed']):
            return 0.3

        # Look for explicit confidence scores
        conf_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        if conf_match:
            return float(conf_match.group(1)) / 100

        # Default medium confidence
        return 0.5

    @staticmethod
    def _classify_claim_type(text: str) -> str:
        """Classify the type of price claim."""
        if any(kw in text for kw in ['return', 'performance', 'gain', 'loss']):
            return 'return'
        elif any(kw in text for kw in ['trend', 'direction', 'momentum']):
            return 'trend'
        elif any(kw in text for kw in ['predict', 'forecast', 'expect', 'next']):
            return 'prediction'
        elif any(kw in text for kw in ['volatility', 'volatile', 'swing']):
            return 'volatility'
        else:
            return 'general'

    @staticmethod
    def _extract_from_structured_sections(report_text: str, report_id: int) -> List[PriceClaim]:
        """Extract claims from tables and structured sections."""
        claims = []

        # Look for markdown tables
        table_pattern = r'\|[^\n]+\|[^\n]+\|'
        tables = re.findall(table_pattern, report_text)

        for table_row in tables:
            # Try to extract horizon, direction, return % from table cells
            cells = [cell.strip() for cell in table_row.split('|') if cell.strip()]

            if len(cells) < 2:
                continue

            # Look for horizon in first cell
            horizon = None
            for cell in cells:
                h = PriceClaimExtractor._extract_horizon(cell.lower())
                if h:
                    horizon = h
                    break

            # Look for return % and direction
            for cell in cells:
                direction = PriceClaimExtractor._extract_direction(cell.lower())
                magnitude = PriceClaimExtractor._extract_magnitude(cell)

                if horizon and direction != 'uncertain' and magnitude != (0.0, 0.0):
                    claims.append(PriceClaim(
                        horizon=horizon,
                        direction=direction,
                        magnitude_min=magnitude[0],
                        magnitude_max=magnitude[1],
                        confidence=0.7,  # Tables generally high confidence
                        source_report_id=report_id,
                        raw_text=table_row.strip(),
                        claim_type='return'
                    ))

        return claims


class ConsensusDetector:
    """Detects consensus across multiple price claims."""

    @staticmethod
    def cluster_claims(claims: List[PriceClaim]) -> Dict[Tuple[str, str], List[PriceClaim]]:
        """
        Cluster claims by (horizon, direction) for consensus analysis.

        Returns:
            Dictionary mapping (horizon, direction) to list of claims
        """
        clusters = defaultdict(list)
        for claim in claims:
            key = (claim.horizon, claim.direction)
            clusters[key].append(claim)
        return dict(clusters)

    @staticmethod
    def calculate_consensus_score(claims: List[PriceClaim], total_reports: int) -> float:
        """
        Calculate consensus score for a cluster of claims.

        Factors:
        - Number of reports supporting (majority agreement)
        - Agreement tightness in magnitude
        - Weighted by individual claim confidence

        Returns:
            Consensus score 0-1
        """
        if not claims:
            return 0.0

        # Factor 1: Report coverage (0-0.5 points)
        unique_reports = len(set(c.source_report_id for c in claims))
        coverage_score = min(unique_reports / total_reports, 1.0) * 0.5

        # Factor 2: Magnitude agreement (0-0.3 points)
        if any(c.magnitude_max > 0 for c in claims):
            magnitudes = [c.magnitude_center() for c in claims if c.magnitude_max > 0]
            if len(magnitudes) > 1:
                magnitude_std = np.std(magnitudes)
                magnitude_mean = np.mean(magnitudes)
                # Coefficient of variation (lower is better)
                cv = magnitude_std / magnitude_mean if magnitude_mean > 0 else 1.0
                agreement_score = max(0, 1 - cv) * 0.3
            else:
                agreement_score = 0.15  # Only one magnitude, partial credit
        else:
            agreement_score = 0.1  # No magnitudes, minimal credit

        # Factor 3: Weighted confidence (0-0.2 points)
        avg_confidence = np.mean([c.confidence for c in claims])
        confidence_score = avg_confidence * 0.2

        return coverage_score + agreement_score + confidence_score

    @staticmethod
    def detect_outliers(claims: List[PriceClaim]) -> List[int]:
        """
        Detect outlier claims based on magnitude.

        Returns:
            List of indices of outlier claims
        """
        if len(claims) < 3:
            return []

        magnitudes = [c.magnitude_center() for c in claims if c.magnitude_max > 0]
        if len(magnitudes) < 3:
            return []

        # Use IQR method
        q1 = np.percentile(magnitudes, 25)
        q3 = np.percentile(magnitudes, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_indices = []
        for i, claim in enumerate(claims):
            if claim.magnitude_max > 0:
                mag = claim.magnitude_center()
                if mag < lower_bound or mag > upper_bound:
                    outlier_indices.append(i)

        return outlier_indices

    @staticmethod
    def identify_disagreements(all_claims: List[PriceClaim], total_reports: int) -> List[Dict]:
        """
        Identify key disagreements where reports conflict.

        Returns:
            List of disagreement descriptions
        """
        disagreements = []

        # Group by horizon
        by_horizon = defaultdict(list)
        for claim in all_claims:
            by_horizon[claim.horizon].append(claim)

        # Check for directional conflicts within each horizon
        for horizon, claims in by_horizon.items():
            directions = [c.direction for c in claims]
            direction_counts = {
                'up': directions.count('up'),
                'down': directions.count('down'),
                'flat': directions.count('flat')
            }

            # Conflict if no clear majority
            total_directional = sum(direction_counts.values())
            if total_directional >= 2:
                max_direction = max(direction_counts.values())
                if max_direction / total_directional < 0.6:  # Less than 60% agreement
                    disagreements.append({
                        'horizon': horizon,
                        'type': 'directional_conflict',
                        'details': direction_counts,
                        'severity': 'high' if max_direction / total_directional < 0.5 else 'medium'
                    })

            # Check for magnitude spread
            magnitudes = [c.magnitude_center() for c in claims if c.magnitude_max > 0]
            if len(magnitudes) >= 2:
                mag_std = np.std(magnitudes)
                mag_mean = np.mean(magnitudes)
                if mag_mean > 0 and mag_std / mag_mean > 0.5:  # High coefficient of variation
                    disagreements.append({
                        'horizon': horizon,
                        'type': 'magnitude_spread',
                        'details': {
                            'mean': round(mag_mean, 2),
                            'std': round(mag_std, 2),
                            'range': [round(min(magnitudes), 2), round(max(magnitudes), 2)]
                        },
                        'severity': 'medium'
                    })

        return disagreements


class PriceSignalSynthesizer:
    """Synthesizes consensus claims into final structured output."""

    @staticmethod
    def generate_consensus_summary(
        consensus_claims: List[Dict],
        disagreements: List[Dict],
        total_reports: int
    ) -> str:
        """Generate natural language summary of consensus."""
        if not consensus_claims:
            return "No strong consensus detected across reports."

        summary_parts = []

        # Group by horizon
        by_horizon = defaultdict(list)
        for claim in consensus_claims:
            by_horizon[claim['horizon']].append(claim)

        # Describe each horizon
        horizon_order = ['next_day', '1_week', '1_month', '3_months', '1_year', '5_years']
        for horizon in horizon_order:
            if horizon in by_horizon:
                claims = by_horizon[horizon]
                # Pick highest consensus claim
                best_claim = max(claims, key=lambda x: x['consensus_score'])

                direction = best_claim['direction']
                mag_range = best_claim['magnitude_range']
                consensus = best_claim['consensus_score']

                horizon_name = horizon.replace('_', ' ').title()

                if mag_range[0] > 0 or mag_range[1] > 0:
                    mag_str = f"{mag_range[0]:.1f}%-{mag_range[1]:.1f}%"
                    summary_parts.append(
                        f"{horizon_name}: {direction} {mag_str} (consensus: {consensus:.0%})"
                    )
                else:
                    summary_parts.append(
                        f"{horizon_name}: {direction} trend (consensus: {consensus:.0%})"
                    )

        summary = "Price consensus across reports: " + "; ".join(summary_parts)

        if disagreements:
            high_severity = [d for d in disagreements if d.get('severity') == 'high']
            if high_severity:
                summary += f". Warning: {len(high_severity)} high-severity disagreement(s) detected."

        return summary

    @staticmethod
    def generate_final_price_signal(consensus_claims: List[Dict]) -> Dict:
        """
        Generate compact final price signal for the trader.

        Prioritizes next_day prediction, falls back to near-term horizons.
        """
        if not consensus_claims:
            return {
                'direction': 'uncertain',
                'expected_pct': 0.0,
                'confidence': 0.0,
                'reasoning': 'Insufficient consensus across reports'
            }

        # Prioritize next_day, then 1_week, then 1_month
        priority_order = ['next_day', '1_week', '1_month', '3_months']

        best_signal = None
        for horizon in priority_order:
            horizon_claims = [c for c in consensus_claims if c['horizon'] == horizon]
            if horizon_claims:
                # Pick claim with highest consensus score
                best_signal = max(horizon_claims, key=lambda x: x['consensus_score'])
                break

        if not best_signal:
            # Fall back to any highest consensus claim
            best_signal = max(consensus_claims, key=lambda x: x['consensus_score'])

        mag_range = best_signal['magnitude_range']
        expected_pct = (mag_range[0] + mag_range[1]) / 2

        return {
            'direction': best_signal['direction'],
            'expected_pct': round(expected_pct, 2),
            'confidence': round(best_signal['consensus_score'], 2),
            'horizon': best_signal['horizon'],
            'reasoning': f"Consensus from {len(best_signal['supporting_reports'])} reports on {best_signal['horizon']} horizon"
        }


def create_price_credibility_scorer(llm, toolkit):
    """
    Create a price credibility scorer node for cross-report consensus analysis.

    This scorer focuses purely on price-derived signals and detects agreement
    across multiple analyst reports.
    """
    model_name = get_model_name(llm)

    def price_credibility_scorer_node(state):
        """Main price credibility scoring logic."""
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        # Get all price reports from state
        price_analyst_state = state.get("price_analyst_state", {})
        price_reports_str = price_analyst_state.get("price_reports_str", "")

        if not price_reports_str or len(price_reports_str.strip()) < 10:
            # Insufficient data for consensus analysis
            return {
                "price_report": "Insufficient price reports for credibility scoring.",
                "price_analyst_state": {
                    "price_reports_str": price_reports_str,
                    "price_credibility_scorer": {
                        "error": "No price reports available"
                    },
                    "count": price_analyst_state.get("count", 0),
                },
            }

        use_llm_invoke = True
        if use_llm_invoke:
            company_info = f"Based only on the provided set of price analysis reports for {company_name} on {trade_date}, perform the following tasks."
            report_info = f"Here is the report: {price_reports_str}"

            context = {
                "role": "user",
                "content": company_info + "\n" + """

## Consider the following price factors:
- **Price trends**: up, down, flat, uncertain.
- **Price ranges**: 0-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50-60%, 60-70%, 70-80%, 80-90%, 90-100%.

## Tasks:

1. High-Confidence Consensus Signals
- Identify price trends and ranges that are consistently reported across multiple price reports and are numerically or semantically aligned (e.g., “price up 10%”, “price down 20%”).
- collect as many high-confidence signals as possible, but don't include any low-confidence signals.

2. Internal Conflicts / Low-Confidence Signals
- Flag any numerical discrepancies, contradictory price trend interpretations, or inconsistent price range descriptions across reports.

3. Evidence Weighting and Ranking
- Select the **high-confidence** and **low-confidence** price trends and ranges
- Return a concise Markdown table for the above 10 facts with columns:
Price Trend | Price Range | Cross-Report Consistency | Confidence Level (High / Medium / Low).

4. Generate **three separate summary reports**:
- **High-Confidence Report**: based only on the high-confidence facts, detail a comprehensive analysis of the price trends and ranges.
- **Low-Confidence Report**: based only on the low-confidence facts, detail a comprehensive analysis of the facts and clearly explaining sources of uncertainty.
- **Data Leakage Audit Report**: State whether data leakage is detected, if yes, list and briefly explain any facts flagged for leakage. If none are found, explicitly state that no data leakage is detected.

5. Decision Making:
- Based on your analysis, provide a specific recommendation and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** X%' where X is the percentage (0-100) of the position to trade. If no specific percentage is mentioned, default to 100%.
- **Output the final decision in the format**: {"action": "BUY|SELL|HOLD", "percentage": number}

""" + report_info
            }

            messages = [
                {
                    "role": "system",
                    "content": f"""You are an experienced Price Credibility Scorer. Evaluate only price trends and ranges derived from historical price data.""",
                },
                context,
            ]

            result = llm.invoke(messages)
            report_str = result.content

            report_dict = extract_report_to_dict(llm, report_str)

            if "high_confidence_report" in report_dict:
                if isinstance(report_dict["high_confidence_report"], str):
                    high_confidence_report = report_dict["high_confidence_report"]
                else:
                    high_confidence_report = report_str
            elif "high_confidence_facts" in report_dict:
                if isinstance(report_dict["high_confidence_facts"], str):
                    high_confidence_report = report_dict["high_confidence_facts"]
                elif isinstance(report_dict["high_confidence_facts"], list):
                    high_confidence_report = "\n".join(report_dict["high_confidence_facts"])
                else:
                    high_confidence_report = report_str
            else:
                high_confidence_report = report_str

            low_confidence_report = None
            if "low_confidence_report" in report_dict and isinstance(report_dict["low_confidence_report"], str):
                low_confidence_report = report_dict["low_confidence_report"]

            decision_high_confidence, decision_low_confidence = extract_decision_from_report(
                llm, high_confidence_report, low_confidence_report
            )
            report_dict["decision_high_confidence"] = decision_high_confidence
            report_dict["decision_low_confidence"] = decision_low_confidence
            report_dict["decision"] = decision_high_confidence
            high_confidence_report = high_confidence_report + f"\n\nFINAL TRANSACTION PROPOSAL: {decision_high_confidence}"
            report_dict["high_confidence_report"] = high_confidence_report
            if low_confidence_report is not None:
                low_confidence_report = low_confidence_report + f"\n\nFINAL TRANSACTION PROPOSAL: {decision_low_confidence}"
                report_dict["low_confidence_report"] = low_confidence_report

            return {
                "price_report": high_confidence_report,
                "price_analyst_state": {
                    "price_reports_str": price_reports_str,
                    "price_credibility_scorer": report_dict,
                    "count": price_analyst_state["count"],
                }
            }


        else:

            # Split reports by the "## Report N from provider:" pattern
            report_pattern = r'Report \d+ from \w+:'
            report_sections = re.split(report_pattern, price_reports_str)
            report_sections = [r.strip() for r in report_sections if r.strip()]

            if len(report_sections) < 1:
                # Single report or unparseable format - return as-is
                return {
                    "price_report": price_reports_str,
                    "price_analyst_state": {
                        "price_reports_str": price_reports_str,
                        "price_credibility_scorer": {
                            "warning": "Only one report available, consensus analysis skipped"
                        },
                        "count": price_analyst_state.get("count", 0),
                    },
                }

            # Extract claims from all reports
            all_claims = []
            for i, report_text in enumerate(report_sections):
                claims = PriceClaimExtractor.extract_claims_from_report(report_text, i)
                all_claims.extend(claims)

            total_reports = len(report_sections)

            # Cluster claims by (horizon, direction)
            claim_clusters = ConsensusDetector.cluster_claims(all_claims)

            # Calculate consensus for each cluster
            consensus_claims = []
            for (horizon, direction), claims in claim_clusters.items():
                consensus_score = ConsensusDetector.calculate_consensus_score(claims, total_reports)

                # Only include if consensus score is above threshold
                if consensus_score >= 0.4:  # Minimum 40% consensus
                    # Filter outliers
                    outlier_indices = ConsensusDetector.detect_outliers(claims)
                    filtered_claims = [c for i, c in enumerate(claims) if i not in outlier_indices]

                    if not filtered_claims:
                        filtered_claims = claims  # Keep all if filtering removes everything

                    # Calculate magnitude range
                    magnitudes = [c.magnitude_center() for c in filtered_claims if c.magnitude_max > 0]
                    if magnitudes:
                        mag_min = min(magnitudes)
                        mag_max = max(magnitudes)
                    else:
                        mag_min = 0.0
                        mag_max = 0.0

                    consensus_claims.append({
                        'horizon': horizon,
                        'direction': direction,
                        'magnitude_range': [round(mag_min, 2), round(mag_max, 2)],
                        'consensus_score': round(consensus_score, 2),
                        'supporting_reports': list(set(c.source_report_id for c in filtered_claims)),
                        'num_claims': len(filtered_claims),
                        'claim_samples': [c.raw_text for c in filtered_claims[:3]]  # Sample claims
                    })

            # Identify disagreements
            disagreements = ConsensusDetector.identify_disagreements(all_claims, total_reports)

            # Generate summary
            consensus_summary = PriceSignalSynthesizer.generate_consensus_summary(
                consensus_claims,
                disagreements,
                total_reports
            )

            # Generate final price signal
            final_price_signal = PriceSignalSynthesizer.generate_final_price_signal(consensus_claims)

            # Build structured output
            result_dict = {
                'trade_date': trade_date,
                'ticker': company_name,
                'total_reports_analyzed': total_reports,
                'total_claims_extracted': len(all_claims),
                'consensus_summary': consensus_summary,
                'consensus_claims': consensus_claims,
                'disagreements': disagreements,
                'final_price_signal': final_price_signal
            }

            # Generate markdown report
            markdown_report = _generate_markdown_report(result_dict)

            # Extract decision from final price signal
            direction_map = {'up': 'BUY', 'down': 'SELL', 'flat': 'HOLD', 'uncertain': 'HOLD'}
            action = direction_map.get(final_price_signal['direction'], 'HOLD')

            # Scale position by confidence
            confidence = final_price_signal['confidence']
            if confidence >= 0.7:
                position_pct = 100
            elif confidence >= 0.5:
                position_pct = 75
            elif confidence >= 0.4:
                position_pct = 50
            else:
                position_pct = 0
                action = 'HOLD'

            decision = {
                'action': action,
                'percentage': position_pct
            }

            result_dict['decision'] = decision
            final_report = markdown_report + f"\n\nFINAL TRANSACTION PROPOSAL: **{action} {position_pct}%**"

            return {
                "price_report": final_report,
                "price_analyst_state": {
                    "price_reports_str": price_reports_str,
                    "price_credibility_scorer": result_dict,
                    "count": price_analyst_state.get("count", 0),
                },
            }

    def price_credibility_scorer_node_load_data_from_cache(state):
        """Load price credibility scorer results from cache."""
        trade_date = state["trade_date"]
        company_name = state["company_of_interest"]

        cache_file = toolkit.config["load_data_from_cache_file"]
        cache_file = cache_file.replace("/model_name/", f"/{model_name}/")
        cache_file = cache_file.replace("/company_name/", f"/{company_name}/")
        cache_file = cache_file.replace("_trade_date_", f"_{trade_date}_")
        cache_file = cache_file + '.json'

        print(f"price_credibility_scorer cache_file: {cache_file}")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data_cache = json.load(f)
        else:
            print(f"Cache file {cache_file} does not exist! Running scorer instead.")
            return price_credibility_scorer_node(state)

        dict_cache = data_cache[trade_date]
        assert company_name.upper() == dict_cache["company_of_interest"].upper(), \
            f"Company {company_name} != {dict_cache['company_of_interest']}"

        if "price_analyst_state" in dict_cache and dict_cache["price_analyst_state"] is not None:
            price_analyst_state = dict_cache["price_analyst_state"]
            price_reports_str = price_analyst_state.get("price_reports_str", "")
            report_dict = price_analyst_state.get("price_credibility_scorer", {})
            count = price_analyst_state.get("count", 0)

            price_report = dict_cache.get("price_report", "")

            print("Loaded price credibility scorer from cache!")

            return {
                "price_report": price_report,
                "price_analyst_state": {
                    "price_reports_str": price_reports_str,
                    "price_credibility_scorer": report_dict,
                    "count": count,
                },
            }
        else:
            print("No price_analyst_state in cache, running scorer instead!")
            return price_credibility_scorer_node(state)

    if toolkit.config.get("load_data_from_cache", False):
        return price_credibility_scorer_node_load_data_from_cache
    else:
        return price_credibility_scorer_node


def _generate_markdown_report(result_dict: Dict) -> str:
    """Generate a markdown report from the consensus analysis results."""
    report = f"""# Price Credibility Score Report

**Trade Date**: {result_dict['trade_date']}
**Ticker**: {result_dict['ticker']}
**Reports Analyzed**: {result_dict['total_reports_analyzed']}
**Claims Extracted**: {result_dict['total_claims_extracted']}

## Consensus Summary

{result_dict['consensus_summary']}

## High-Confidence Consensus Claims

"""

    if result_dict['consensus_claims']:
        report += "| Horizon | Direction | Magnitude Range | Consensus Score | Supporting Reports |\n"
        report += "|---------|-----------|-----------------|-----------------|--------------------|\n"

        for claim in sorted(result_dict['consensus_claims'], key=lambda x: -x['consensus_score']):
            horizon_name = claim['horizon'].replace('_', ' ').title()
            mag_str = f"{claim['magnitude_range'][0]:.1f}% - {claim['magnitude_range'][1]:.1f}%"
            if claim['magnitude_range'][0] == 0 and claim['magnitude_range'][1] == 0:
                mag_str = "N/A"

            report += f"| {horizon_name} | {claim['direction'].upper()} | {mag_str} | "
            report += f"{claim['consensus_score']:.0%} | {len(claim['supporting_reports'])}/{result_dict['total_reports_analyzed']} |\n"

        report += "\n### Sample Supporting Evidence\n\n"
        for claim in result_dict['consensus_claims'][:3]:  # Top 3
            report += f"**{claim['horizon'].replace('_', ' ').title()} - {claim['direction'].upper()}**:\n"
            for sample in claim['claim_samples']:
                report += f"- {sample}\n"
            report += "\n"
    else:
        report += "*No high-confidence consensus detected.*\n\n"

    report += "## Disagreements and Conflicts\n\n"

    if result_dict['disagreements']:
        for disagreement in result_dict['disagreements']:
            horizon_name = disagreement['horizon'].replace('_', ' ').title()
            report += f"- **{horizon_name} - {disagreement['type'].replace('_', ' ').title()}** "
            report += f"(Severity: {disagreement['severity']})\n"
            report += f"  - Details: {disagreement['details']}\n\n"
    else:
        report += "*No significant disagreements detected.*\n\n"

    report += "## Final Price Signal\n\n"
    signal = result_dict['final_price_signal']
    report += f"- **Direction**: {signal['direction'].upper()}\n"
    report += f"- **Expected Change**: {signal['expected_pct']}%\n"
    report += f"- **Confidence**: {signal['confidence']:.0%}\n"
    report += f"- **Horizon**: {signal.get('horizon', 'N/A').replace('_', ' ').title()}\n"
    report += f"- **Reasoning**: {signal['reasoning']}\n"

    return report