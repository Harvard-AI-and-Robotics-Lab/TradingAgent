# TradingAgents/graph/signal_processing.py
import json
from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision with percentage.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY X%, SELL X%, or HOLD)
        """
        messages = [
            (
                "system",
                """You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts.
Your task is to extract the investment decision with percentage: SELL X%, BUY X%, or HOLD.
If no specific percentage is mentioned, default to 100%.
Provide only the extracted decision in the format "ACTION X%" (e.g., "BUY 100%", "SELL 50%", "HOLD"), without adding any additional text or information.""",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content

    def process_signal_with_percentage(self, full_signal) -> dict:
        """
        Process a trading signal to extract decision and percentage only.

        Args:
            full_signal: Complete trading signal text (e.g., "sell 50% stocks" or "buy 100%") or dict

        Returns:
            Dictionary with 'action' (BUY/SELL/HOLD) and 'percentage' (0-100)
        """
        # If already a dict, validate and return
        if isinstance(full_signal, dict):
            if 'action' in full_signal and 'percentage' in full_signal:
                return full_signal
            else:
                # Convert dict to string for processing
                full_signal = str(full_signal)

        # Ensure full_signal is a string
        if not isinstance(full_signal, str):
            full_signal = str(full_signal)

        messages = [
            (
                "system",
                """You are an assistant that extracts trading decisions from text.

Extract the action (BUY, SELL, or HOLD) and the percentage (0-100).

Rules:
1. If the text explicitly mentions a percentage (e.g., "buy 25%", "sell half"), use that number.
2. If no specific percentage is mentioned:
   - For BUY or SELL, default to 100.
   - For HOLD, use the **current position percentage** from context (do not default to 0).
3. "HOLD" means maintaining the existing position — not going to cash.
4. Only when no clear BUY or SELL intent appears, the action should be "HOLD".
5. Return ONLY a JSON object in the exact format:
   {"action": "BUY|SELL|HOLD", "percentage": <number>}
6. Note the distinction between the percentage of **price fluctuation** and **the percentage representing investment recommendations (buy/sell/hold)**.

Examples:
- "sell 50% stocks" -> {"action": "SELL", "percentage": 50}
- "sell now" -> {"action": "SELL", "percentage": 100}
- "buy the stock" -> {"action": "BUY", "percentage": 100}
- "buy 25%" -> {"action": "BUY", "percentage": 25}
- "hold position" -> {"action": "HOLD", "percentage": current_position}
- "maintain 25% allocation" -> {"action": "HOLD", "percentage": 25}
- "sell half" -> {"action": "SELL", "percentage": 50}
- "allocating **50%**" -> {"action": "BUY", "percentage": 50}


Output the final decision in the format:
{"action": "BUY|SELL|HOLD", "percentage": <number>}
""",
            ),
            ("human", full_signal),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        response = extract_decision_from_string_to_json(response)
        return response


def extract_decision_from_string_to_json(text: str):
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    s = text.strip()

    if s.startswith("```"):
        s = s.strip("`")
        # ```json\n...\n```
        lines = s.splitlines()
        if lines and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:]).strip()

    # find the first { or [
    start_candidates = [i for i in [s.find("{")] if i != -1]
    if not start_candidates:
        # raise ValueError(f"No JSON object/array found in response: {s[:200]!r}")
        print(f"No JSON object/array found in response: {s[:200]!r}")
        return {
            "action": "HOLD",
            "percentage": 0,
            "Note": f"No JSON object/array found in response: {s[:200]!r}"
        }

    start = min(start_candidates)

    # last }
    end = s.rfind("}")

    if end < start:
        # raise ValueError(f"Malformed JSON boundaries: {s[:200]!r}")
        print(f"No JSON object/array found in response: {s[:200]!r}")
        return {
            "action": "HOLD",
            "percentage": 0,
            "Note": f"Malformed JSON boundaries: {s[:200]!r}"
        }

    json_str = s[start:end+1].strip()

    try:
        dict_result = json.loads(json_str)

        if not isinstance(dict_result, dict):
            return {
                "action": "HOLD",
                "percentage": 0,
                "Note": f"Not a dictionary, json_str: {json_str}"
            }

        return {
            "action": dict_result.get("action", "HOLD"),
            "percentage": dict_result.get("percentage", 0),
        }
    except Exception as e:
        # print(f"Error parsing JSON: {e}, json_str: {json_str}")
        return {
            "action": "HOLD",
            "percentage": 0,
            "Note": f"Error parsing JSON: {e}, json_str: {json_str}"
        }