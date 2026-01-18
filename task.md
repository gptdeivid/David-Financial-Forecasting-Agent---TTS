Task: Financial Forecasting Agent for TCS

Your task is to build a FastAPI application that acts as an AI agent capable of generating a business outlook forecast for Tata Consultancy Services (TCS).
The agent's primary function is to move beyond simple Q&A. It must automatically find and analyze financial documents from past 1-2 quarters to generate a reasoned, qualitative forecast for the future.
Source: You are expected to be resourceful. Find and download the necessary documents (e.g., quarterly financial reports, earnings call transcripts) for the last 1-2 quarters from a source like <https://www.screener.in/company/TCS/consolidated/#documents>
Usage of AI
At Elevation AI, we embrace AI-first solutions. For this assignment, if you have used AI, we’re keen to understand how. Please document:
Your AI stack used and reasoning approach
The specific tools/models employed (e.g., OCR, RAG stack, embeddings, vector DB, LLM provider, function-calling).
What the AI actually achieved end-to-end (data sources retrieved, metrics extracted, synthesis quality).
Guardrails and evaluation (prompting strategy, retries, grounding checks).
Limits and tradeoffs you encountered—and how you mitigated them.
Core Requirements
You will build an agent with access to at least two specialized, purpose-built tools:
FinancialDataExtractorTool: A robust tool designed to understand quarterly financial reports and extract key financial metrics (e.g., Total Revenue, Net Profit, Operating Margin).

QualitativeAnalysisTool: A RAG-based tool that performs semantic search and analysis across 2-3 past earnings call transcripts to identify recurring themes, management sentiment, and forward-looking statements.
Deliverables
Generate a Forecast: The primary endpoint of your API must be able to handle a complex analytical task.
Example Task: "Analyze the financial reports and transcripts for the last three quarters and provide a qualitative forecast for the upcoming quarter. Your forecast must identify key financial trends (e.g., revenue growth, margin pressure), summarize management's stated outlook, and highlight any significant risks or opportunities mentioned."

Provide Structured Output: The agent's final output must be a structured JSON object. This demonstrates your ability to control the LLM and deliver predictable, machine-readable results.

Log the Results: The agent must be served via a FastAPI endpoint, and all incoming requests and the final JSON output must be logged to a MySQL database.
Optional, not Necessary
MarketDataTool: As an optional bonus, you can implement a third tool that fetches live market data (e.g., current stock price) and incorporates it as another point of context in the analysis.
Technical Stack & Expectations
Programming Language: Python 3.10+
Backend Framework: FastAPI
LLM Framework: LangChain
AI Provider: Any
Database: MySQL 8.0
What to Submit & The Importance of the README
Your submission will be evaluated not just on the code, but on how easy it is for us to understand and run. Another engineer must be able to clone your repository, follow your instructions, and have the service running locally without any guesswork.
Please provide a link to a Git repository containing:
Source Code: All your Python scripts.
requirements.txt: A file listing all necessary libraries.
README.md: This must include:
Project Overview: Your architectural approach, design choices, and how your agent chains thoughts and tools to create a forecast.
Agent & Tool Design: A detailed explanation of each tool and the master prompt you used to guide your agent's reasoning.
Setup Instructions: Clear, step-by-step instructions on setting up the environment, installing dependencies, and configuring all credentials (LLMs and MySQL). This must be unambiguous.
How to Run: The exact commands to start the FastAPI service.
How We Will Evaluate Your Submission
Reasoning: Does the agent successfully perform a multi-step analysis? Can it synthesize data from multiple documents and tools into a coherent forecast?
Engineering & Architecture: How well-designed are your tools and agentic chain? Is the logic for extracting financial data robust?
Code Quality & Readability: Is your code clean, modular, and easy to maintain? Does it follow best practices for a production-ready service?
Clarity and Reproducibility of Documentation: Can we run your project just by reading your README? How clear are your explanations of your design?
