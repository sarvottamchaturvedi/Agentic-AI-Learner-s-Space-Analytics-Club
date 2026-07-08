import json
import os
import requests
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Literal
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in the .env file")

# Initialize the genai Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"


# ==========================================
# STATE DEFINITION & DATA STRUCTURES
# ==========================================

class SecurityImpact(TypedDict):
    ticker: str
    bias: Literal["Long", "Short", "Neutral"]
    expected_volatility_spike: str
    rationale: str
    source_citation: str


class MarketIntelligenceState(TypedDict):
    search_topic: str
    raw_news: str
    news_source: str
    macro_vectors: Dict[str, str]
    first_order_targets: List[SecurityImpact]
    risk_assessment: str
    second_order_critique: str
    aggregator_verdict: str
    consensus_reached: bool
    iteration_count: int


# ==========================================
# DEFINING THE SPECIALIZED AGENTS
# ==========================================

def fetch_live_news_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 0: Autonomous News Discoverer & Fetcher
    1. Grabs the top global business headlines.
    2. Uses Gemini to identify the most disruptive single macro event.
    3. Queries NewsAPI for deep coverage of that specific event.
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not found in the .env file")

    # Discover current top headlines
    top_url = "https://newsapi.org/v2/top-headlines"
    top_params = {
        "category": "business",
        "language": "en",
        "pageSize": 15,
        "apiKey": NEWS_API_KEY
    }

    print("[Agent 0] Scanning global business headlines for the biggest macro disruption...")
    top_response = requests.get(top_url, params=top_params).json()

    headlines = [article['title'] for article in top_response.get("articles", []) if article.get('title')]
    headlines_text = "\n".join(headlines)

    # Use Gemini to pick the biggest theme
    discovery_prompt = f"""
    Here are the current top business headlines:
    {headlines_text}

    Identify the single most market-disruptive macro or geopolitical event from this list.
    Output ONLY a concise 4-8 word search query that captures this specific event to search a news database.
    Do not use quotes, explanations, or introductory text.
    """

    discovery_response = client.models.generate_content(
        model=MODEL_ID,
        contents=discovery_prompt
    )

    dynamic_topic = discovery_response.text.strip().replace('"', '')
    print(f"[Agent 0] Auto-discovered trending topic: '{dynamic_topic}'\n")

    # Fetch deep coverage on this specific topic
    search_url = "https://newsapi.org/v2/everything"
    search_params = {
        "q": dynamic_topic,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 3,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(search_url, params=search_params)
    data = response.json()

    if data.get("status") != "ok":
        return {
            "search_topic": dynamic_topic,
            "raw_news": f"Failed to retrieve news: {data.get('message', 'Unknown Error')}",
            "news_source": "API Error Pipeline"
        }

    articles = data.get("articles", [])
    if not articles:
        return {
            "search_topic": dynamic_topic,
            "raw_news": "No recent breaking news articles found for this specific macro vector.",
            "news_source": "Null Search Stream"
        }

    aggregated_news = ""
    sources_found = []

    for i, article in enumerate(articles):
        src = article["source"]["name"]
        sources_found.append(src)
        aggregated_news += f"--- Macro Article {i + 1} [{src}] ---\n"
        aggregated_news += f"Title: {article['title']}\n"
        aggregated_news += f"Snippet: {article['description']}\n\n"

    return {
        "search_topic": dynamic_topic,
        "raw_news": aggregated_news.strip(),
        "news_source": ", ".join(list(set(sources_found)))
    }

def ingest_macro_news_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 1: Ingestor & Parser
    Extracts the fundamental vectors from the news and standardizes citations.
    """
    prompt = f"""
    You are a Macro Research Ingestor. Analyze the following news item:
    Source: {state['news_source']}
    Content: {state['raw_news']}

    Extract the fundamental macro vectors in JSON format. Provide these exact keys:
    - primary_event_type (e.g., Geopolitical Conflict, Supply Chain, Monetary Policy)
    - geographic_epicenter
    - anticipated_time_horizon (e.g., Immediate, 1-3 Months)
    - key_catalyst_summary
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )

    vectors = json.loads(response.text)
    return {"macro_vectors": vectors, "iteration_count": 0}


def analyze_direct_impact_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 2: Alpha Generator (First-Order Analyst)
    Identifies immediate directional biases, specific tickers, and expected volatility.
    """
    feedback_context = ""
    if state.get("aggregator_verdict") and not state.get("consensus_reached"):
        feedback_context = f"""
        CRITICAL REJECTION FROM PREVIOUS ITERATION:
        Your previous target securities were rejected. Incorporate this feedback to refine your picks:
        - Risk Assessment Feedback: {state.get('risk_assessment')}
        - Second-Order Critique: {state.get('second_order_critique')}
        """

    prompt = f"""
    You are an Alpha Generator specializing in first-order market responses.
    Macro Vectors: {json.dumps(state['macro_vectors'])}
    Source Material: {state['raw_news']}
    {feedback_context}

    Identify 1 to 3 specific securities (orders, structural tickers, or commodities) that will experience immediate directional price action.
    Provide your reasoning and assign an expected volatility spike description (e.g., Extreme Spike, High Volatility, Moderate).

    You MUST respond with a JSON object matching this exact schema:
    {{
        "securities": [
            {{
                "ticker": "TICKER_OR_ASSET",
                "bias": "Long" or "Short" or "Neutral",
                "expected_volatility_spike": "volatility description",
                "rationale": "detailed logic based on first principles",
                "source_citation": "{state['news_source']}"
            }}
        ]
    }}
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )

    data = json.loads(response.text)
    return {"first_order_targets": data.get("securities", [])}


def risk_mathematical_filter_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 3: Quantitative Risk Manager
    Evaluates the math, risk/reward profile, and volatility viability of Agent 2's ideas.
    """
    prompt = f"""
    You are a Quantitative Risk Manager. Evaluate if the proposed trades by the Alpha Generator are mathematically sound or too dangerous given the macro volatility.
    Proposed Trades: {json.dumps(state['first_order_targets'])}
    Macro Context: {json.dumps(state['macro_vectors'])}

    Analyze if the expected volatility spike justifies the directional bias. Look for tail-risk hazards.
    Provide a concise, brutal evaluation. State clearly if you APPROVE or REJECT the risk metrics of these targets.
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return {"risk_assessment": response.text}


def analyze_second_order_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 4: Macro Second-Order Critic
    Looks for hidden downstream effects, capital rotation, or demand destruction.
    """
    prompt = f"""
    You are a cynical Macro Economist and Second-Order Critic. Look beyond the obvious first-order targets.
    Proposed Trades: {json.dumps(state['first_order_targets'])}
    Original Event: {state['raw_news']}

    Critique the Alpha Generator's thesis by identifying secondary systemic chain reactions.
    State clearly if you find gaps in their logic or if you APPROVE.
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return {"second_order_critique": response.text}


def consensus_aggregator_node(state: MarketIntelligenceState) -> Dict:
    """
    Agent 5: The System Aggregator
    Synthesizes critiques and decides whether to route back or output.
    """
    current_iteration = state.get("iteration_count", 0) + 1

    prompt = f"""
    You are the Lead Investment Officer. Review the entire debate between your agents:
    - First-Order Trades: {json.dumps(state['first_order_targets'])}
    - Risk Team Assessment: {state['risk_assessment']}
    - Macro Critic Assessment: {state['second_order_critique']}

    Are the proposed trades fundamentally sound, properly cited, and safe to execute according to both critiques? 
    If BOTH agents generally approve, or if you can reconcile their slight differences into a final bulletproof thesis, mark consensus as TRUE.
    If there are massive flaws, unmitigated risks, or missing systemic perspectives, mark consensus as FALSE.

    Respond strictly with a JSON object containing these exact keys:
    {{
        "consensus_approved": true or false,
        "final_verdict_summary": "A synthesized final master report if approved, or detailed instructions to rewrite if false."
    }}
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )

    decision = json.loads(response.text)
    is_approved = decision.get("consensus_approved", False)

    # Prevent endless loops
    if current_iteration >= 3:
        is_approved = True

    return {
        "consensus_reached": is_approved,
        "aggregator_verdict": decision.get("final_verdict_summary", ""),
        "iteration_count": current_iteration
    }


# ==========================================
# ORCHESTRATION PIPELINE USING LANGGRAPH
# ==========================================

workflow = StateGraph(MarketIntelligenceState)

workflow.add_node("fetch_news", fetch_live_news_node)
workflow.add_node("ingest_news", ingest_macro_news_node)
workflow.add_node("alpha_generator", analyze_direct_impact_node)
workflow.add_node("risk_manager", risk_mathematical_filter_node)
workflow.add_node("macro_critic", analyze_second_order_node)
workflow.add_node("aggregator", consensus_aggregator_node)

workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "ingest_news")
workflow.add_edge("ingest_news", "alpha_generator")

workflow.add_edge("alpha_generator", "risk_manager")
workflow.add_edge("alpha_generator", "macro_critic")

workflow.add_edge("risk_manager", "aggregator")
workflow.add_edge("macro_critic", "aggregator")


def route_based_on_consensus(state: MarketIntelligenceState):
    if state["consensus_reached"]:
        return END
    else:
        print(f"\n🔄 [DEBUG] Consensus failing at iteration {state['iteration_count']}. Prompting loop rewrite...")
        return "alpha_generator"


workflow.add_conditional_edges("aggregator", route_based_on_consensus)
compiled_market_system = workflow.compile()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    live_macro_query = {
        "search_topic": ""
    }

    print("Running Fully Autonomous Market Intelligence Agent...\n")
    final_state = compiled_market_system.invoke(live_macro_query)

    print("\n" + "=" * 60)
    print("FINAL CONSOLIDATED MARKET INTELLIGENCE REPORT")
    print("=" * 60)
    print(f"Total Framework Iterations: {final_state['iteration_count']}\n")

    print("AGGREGATED API SOURCE LOGS:")
    print(f"Sources Sampled: {final_state['news_source']}\n")

    print("TARGET SECURITIES POOL:")
    for target in final_state['first_order_targets']:
        print(f" - Asset/Ticker: {target['ticker']}")
        print(f"   Bias:         {target['bias']}")
        print(f"   Velocity/Vol: {target['expected_volatility_spike']}")
        print(f"   Source Claim: [{target['source_citation']}]")
        print(f"   Rationale:    {target['rationale']}\n")

    print("LEAD VERDICT & EXECUTIVE SUMMARY:")
    print(final_state['aggregator_verdict'])