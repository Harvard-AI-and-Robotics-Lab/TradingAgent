import json
import re
from typing import Any, Dict

from tradingagents.graph.signal_processing import extract_decision_from_string_to_json


def get_model_name(llm):
    for attr in ["model", "model_name", "model_id"]:
        if hasattr(llm, attr):
            val = getattr(llm, attr)
            if isinstance(val, str):
                return val
    return None

def extract_json_block(text: str) -> str:
    """
    Extract JSON from a response that might include code fences.
    """
    text = text.strip()
    # Remove ```json fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_report_to_dict(llm, report_text: str) -> Dict[str, Any]:
    """
    Uses an LLM to convert a markdown report into a structured dict (strict JSON).
    llm: another LLM instance (can be the same) used purely for extraction.
    """
    schema_hint = {
        "trade_date": "YYYY-MM-DD or null",
        "ticker": "string or null",
        "high_confidence_facts": ["string", "..."],
        "low_confidence_signals": ["string", "..."],
        "evidence_table": ["string", "..."],
        "high_confidence_report": "string",
        "low_confidence_report": "string",
        "data_leakage_audit": {
            "leakage_detected": True,
            "leakage_report": "string",
        },
        "decision": "string",
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction engine.\n"
                "Convert the provided markdown report into STRICT JSON that matches the given schema.\n"
                "Rules:\n"
                "- Output JSON only.\n"
                "- Use null when missing.\n"
                "- Preserve wording of facts as-is.\n"
                "- evidence_table must be parsed from the markdown table if present.\n"
                "- leakage_detected must be true if the report explicitly states leakage is detected.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Schema (example types):\n{json.dumps(schema_hint, indent=2)}\n\n"
                f"Markdown report:\n{report_text}"
            )
        }
    ]

    resp = llm.invoke(messages)
    raw = extract_json_block(resp.content)

    try:
        report_dict = json.loads(raw)
        if "high_confidence_report" in report_dict and "high_confidence_facts" in report_dict:
            report_dict["high_confidence_report"] += "\n\nhigh_confidence_facts:\n" + "\n".join(report_dict["high_confidence_facts"])
        if "low_confidence_report" in report_dict and "low_confidence_signals" in report_dict:
            report_dict["low_confidence_report"] += "\n\nlow_confidence_signals:\n" + "\n".join(report_dict["low_confidence_signals"])
        return report_dict
    except json.JSONDecodeError:
        # Fallback: ask LLM to repair JSON
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Fix the provided content into valid STRICT JSON.\n"
                    "Rules:\n"
                    "- Output JSON only.\n"
                    "- Do not change values unless required to make JSON valid.\n"
                )
            },
            {"role": "user", "content": raw}
        ]
        repaired = llm.invoke(repair_messages)
        repaired_raw = extract_json_block(repaired.content)
        return json.loads(repaired_raw)


def extract_decision_from_report(llm, high_confidence_report: str, low_confidence_report: str) -> tuple[dict, dict]:

    if high_confidence_report is None:
        high_confidence_report = ""
    if low_confidence_report is None:
        low_confidence_report = ""

    if high_confidence_report != "":
        # get decision from high_confidence_report
        messages = [
            {
                "role": "system",
                "content": f"""You are an experienced Fundamental Analyst. Extract the final decision from the provided report.s""",
            },
            {
                "role": "user",
                "content": high_confidence_report + "\n" + """
                **Output the final decision in the format**: {"action": "BUY|SELL|HOLD", "percentage": number}
                """
        }
        ]
        decision_high_confidence = llm.invoke(messages).content
        decision_high_confidence = extract_decision_from_string_to_json(decision_high_confidence)
    else:
        decision_high_confidence = None

    if low_confidence_report != "":
        # get decision from low_confidence_report
        messages = [
            {
                "role": "system",
                "content": f"""You are an experienced Fundamental Analyst. Extract the final decision from the provided report.s""",
            },
            {
                "role": "user",
                "content": low_confidence_report + "\n" + """
                **Output the final decision in the format**: {"action": "BUY|SELL|HOLD", "percentage": number}
                """
        }
        ]
        decision_low_confidence = llm.invoke(messages).content
        decision_low_confidence = extract_decision_from_string_to_json(decision_low_confidence)
    else:
        decision_low_confidence = None

    return decision_high_confidence, decision_low_confidence