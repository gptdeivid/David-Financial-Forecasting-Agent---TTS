# 📊 Financial Forecasting Agent for TCS

## 🎯 Task Overview

Build a **FastAPI** application that acts as an AI agent capable of generating a business outlook forecast for **Tata Consultancy Services (TCS)**.

The agent's primary function is to move beyond simple Q&A. It must automatically find and analyze financial documents from the past 1-2 quarters to generate a reasoned, qualitative forecast for the future.

---

## 🗂️ Source Material

You are expected to be resourceful. Find and download the necessary documents for the last 1-2 quarters from sources such as:

* [Screener.in - TCS Consolidated Documents](https://www.screener.in/company/TCS/consolidated/#documents)
* *e.g., Quarterly financial reports, earnings call transcripts*

---

## 🤖 Usage of AI

At Elevation AI, we embrace AI-first solutions. If you use AI for this assignment, please document:

* **AI Stack:** Your chosen tools and reasoning approach.
* **Specific Tools/Models:** OCR, RAG stack, embeddings, vector DB, LLM provider, function-calling, etc.
* **End-to-End Achievement:** Data sources retrieved, metrics extracted, synthesis quality.
* **Guardrails & Evaluation:** Prompting strategy, retries, grounding checks.
* **Limits & Tradeoffs:** Challenges encountered and how they were mitigated.

---

## ⚙️ Core Requirements

Build an agent with access to at least two specialized, purpose-built tools:

1. **FinancialDataExtractorTool**: Extracts key financial metrics (e.g., Total Revenue, Net Profit, Operating Margin) from quarterly financial reports.
2. **QualitativeAnalysisTool**: A RAG-based tool for semantic search and analysis across 2-3 past earnings call transcripts to identify recurring themes, management sentiment, and forward-looking statements.

### ➕ Optional (Bonus)

* **MarketDataTool**: Fetch live market data (e.g., current stock price) and incorporate it as additional context.

---

## 📦 Deliverables

### 1. Generate a Forecast

The primary endpoint of your API must handle a complex analytical task.
> **Example Task:** "Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter. Your forecast must identify key financial trends (e.g., revenue growth, margin pressure), summarize management's stated outlook, and highlight any significant risks or opportunities mentioned."

### 2. Structured Output

The agent's final output must be a structured **JSON object** to ensure predictable, machine-readable results.

### 3. Log the Results

All incoming requests and the final JSON output must be logged to a **MySQL database**.

---

## 🛠️ Technical Stack & Expectations

| Component | Requirement |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Framework** | FastAPI |
| **LLM Framework** | LangChain |
| **AI Provider** | Any |
| **Database** | MySQL 8.0 |

---

## 📄 Submission & Documentation

Another engineer must be able to clone your repository and have the service running locally without guesswork.

### Repository Contents

1. **Source Code**: All Python scripts.
2. **`requirements.txt`**: All necessary libraries.
3. **`README.md`** (Crucial):
    * **Project Overview**: Architectural approach, design choices, and agent reasoning chain.
    * **Agent & Tool Design**: Explanation of each tool and the master prompt used.
    * **Setup Instructions**: Step-by-step environment setup, dependency installation, and credential configuration (LLMs and MySQL).
    * **How to Run**: Exact commands to start the FastAPI service.

---

## 🏆 Evaluation Criteria

* **Reasoning**: Ability to synthesize data from multiple sources into a coherent forecast.
* **Engineering & Architecture**: Design of tools and agentic chains.
* **Code Quality**: Modular, clean, and production-ready code.
* **Clarity & Reproducibility**: Quality of documentation and ease of local setup.
