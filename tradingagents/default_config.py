import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache"
    ),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache/market_data/price_data",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url":  "https://api.openai.com/v1",  # Official OpenAI API
    "analyst_llms": [],
    "analyst_llm_providers": [],
    "credibility_scorer_llm": None,
    "credibility_scorer_llm_provider": None,
    # "analyst_llms": ["gpt-4o-mini", "gemini-2.5-flash-lite"],
    # "analyst_llm_providers": ["openai", "google"],
    # "credibility_scorer_llm": "grok-4-fast-reasoning",
    # "credibility_scorer_llm_provider": "xai",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 20,  # 100
    # Tool settings
    "online_tools": True,
    "resume_from_cache": False,
    "load_data_from_cache": False,
    "load_data_from_cache_file": "",
    # Reflective memory settings
    "return_periods": [1, 7, 14, 28, 90, 180, 360],
}
