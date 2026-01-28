# outcome_explainer.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """Je bent een scherpe Retail Performance Analyst voor een regiomanager.
Schrijf in het Nederlands. Wees concreet, actiegericht en kort. Geen managementfluff.

Output EXACT in deze structuur:
1) Samenvatting (max 5 bullets)
2) Belangrijkste drivers (max 5 bullets)
3) Acties (3–7 bullets, elk met: wat / waarom / hoe je succes meet)
4) Vragen om de analyse te verscherpen (max 3 bullets)

Regels:
- Gebruik cijfers uit de input waar mogelijk.
- Als data ontbreekt: benoem dat expliciet en doe een beperkte aanname (1 zin).
- Verzin geen KPI’s die niet in de input zitten.
"""


def _get_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY ontbreekt. Zet deze in Streamlit Secrets.")
    return OpenAI(api_key=key)


def explain_region_brief(
    kpis: Dict[str, Any],
    opportunities: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.25,
) -> str:
    """
    Maakt een regiomanager-brief op basis van:
    - region-kpis (rollup)
    - top opportunities (per store)
    - optionele context (bv. macro/benchmarks/notes)

    Verwacht dat opportunities al 'presentable' zijn (namen, KPI’s, score, etc.).
    """
    client = _get_client(api_key)
    m = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    payload = {
        "region_kpis": kpis,
        "top_opportunities": opportunities,
        "context": context or {},
        "instructions": "Focus op impact, quick wins en prioritering voor een regiomanager.",
    }

    # compact, maar traceerbaar
    user_content = "Input (JSON):\n" + json.dumps(payload, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=m,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()