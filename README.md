# Multi-Agent System for News and Market Analysis

## Overview
This repository contains an autonomous, multi-agent AI system designed to fetch, ingest, and analyze live macroeconomic news to generate institutional-grade market intelligence. 

This repository serves as my official project submission for the **Learners' Space 2026 Capstone Project**. Completing this working multi-agent architecture satisfies the compulsory capstone requirement for the course certification.

---

## Problem Statement
Financial markets react to geopolitical and macroeconomic news in milliseconds. However, immediate first-order reactions (e.g., "war means oil goes up") are often dangerous traps that ignore liquidity crunches, demand destruction, and systemic tail risks. 

This project tackles a complex financial problem that genuinely requires multiple agents coordinating, rather than just a single agent with extra steps. By forcing a generative "Alpha" agent to defend its trading thesis against two specialized, parallel "Risk" and "Macro" critique agents, the system simulates the rigorous debate of a real hedge fund investment committee.

---

## Orchestration Design
This system utilizes the **Parallel + Aggregator orchestration pattern** to coordinate its workflow. Built using **LangGraph**, **NewsAPI** and the **Google Gemini SDK** (`gemini-2.5-flash`), the architecture features 6 distinct components with a clear division of responsibility:

```
[Agent 0: News Discoverer] 
           │
           ▼
[Agent 1: Ingestor & Parser] 
           │
           ▼
[Agent 2: Alpha Generator] 
      /          \
     /            \  (Parallel Analysis)
    ▼              ▼
[Agent 3: Risk]  [Agent 4: Critic]
    \              /
     \            /  (Join / Fan-in)
      ▼          ▼
[Agent 5: System Aggregator] ──(Fails Consensus?)──► Loop back to Agent 2
           │
    (Consensus/Max Iter)
           │
           ▼
     [Final Report]
```

*   **Agent 0: Autonomous News Discoverer** – Scans global business headlines, autonomously identifies the most disruptive macro event, and fetches deep-dive coverage via NewsAPI.
*   **Agent 1: Ingestor & Parser** – Extracts fundamental macro vectors (event type, epicenter, time horizon) into a standardized JSON schema.
*   **Agent 2: Alpha Generator** – Acts as the first-order analyst, proposing directional trades (Long/Short), target tickers, and expected volatility based on the news.
*   **Agent 3: Quantitative Risk Manager** – Runs in parallel to evaluate the mathematical viability, risk/reward profiles, and tail-risk hazards of the proposed trades.
*   **Agent 4: Macro Second-Order Critic** – Runs in parallel to look beyond the obvious, identifying hidden downstream effects, capital rotation, or demand destruction.
*   **Agent 5: System Aggregator** – Acts as the Lead Investment Officer. It synthesizes the parallel critiques and determines if consensus is reached. If the trades are too risky, it rejects them and forces Agent 2 into a loop with corrective feedback (capped at 3 iterations via a circuit breaker).

---

## Technical Execution & Setup

### Prerequisites
*   Python 3.10+
*   Google Gemini API Key
*   NewsAPI Key

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sarvottamchaturvedi/Agentic-AI-Learner-s-Space-Analytics-Club.git](https://github.com/sarvottamchaturvedi/Agentic-AI-Learner-s-Space-Analytics-Club.git)
    cd Agentic-AI-Learner-s-Space-Analytics-Club
    ```
2.  **Install dependencies:**
    ```bash
    pip install google-genai langgraph python-dotenv requests
    ```
3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GEMINI_API_KEY=your_gemini_key_here
    NEWS_API_KEY=your_newsapi_key_here
    ```

### Running the System
Execute the pipeline via your terminal:
```bash
python agent.py
```

---

## 🏆 Acknowledgements
This project was built as part of the **Agentic AI** track for the **Learners' Space 2026** bootcamp[cite: 1]. Special thanks to the **Analytics Club** team for the mentorship, theory frameworks, and guidance throughout the cohort[cite: 1].
