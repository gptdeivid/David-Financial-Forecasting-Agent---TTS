# Comprehensive Development Guide: TCS Business Outlook Forecast Agent

## Executive Checklist: What You Will Do

This guide outlines the complete development of a FastAPI-powered AI agent for TCS business forecasting. Here's what you'll accomplish:

- **Design & architect** a multi-tool agent system with specialized financial and qualitative analysis components
- **Implement document acquisition** workflow to fetch and ingest TCS financial reports from Screener.in
- **Build RAG pipeline** with embeddings, vector storage, and semantic search for earnings call analysis
- **Create LangChain agent** with function calling to orchestrate tool invocations for compound reasoning
- **Expose FastAPI endpoints** with async request handling and structured JSON responses
- **Establish audit trail** via MySQL 8.0 logging of all requests and AI reasoning artifacts

---

# 1. Project Overview

## Problem Statement

Investors, analysts, and corporate planners need rapid, data-driven business outlook forecasts for Tata Consultancy Services (TCS). Manual analysis of quarterly reports, earnings calls, and market data is time-consuming and prone to inconsistency.

## Solution Approach

Build an **AI-first agent** that:

1. **Automatically retrieves** recent financial documents (10-Qs, earnings transcripts) from public sources
2. **Extracts quantitative insights** (revenue, margins, growth rates) via LLM-powered document analysis
3. **Analyzes qualitative trends** through RAG-based semantic search of earnings calls
4. **Synthesizes forecasts** by reasoning across both quantitative and qualitative signals
5. **Logs audit trail** to MySQL for governance and reproducibility

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LangChain Agent (Orchestrator)          │   │
│  │  - Reason about financial forecast task             │   │
│  │  - Call tools sequentially or in parallel           │   │
│  │  - Aggregate results into forecast                  │   │
│  └──────────────┬───────────────────────┬──────────────┘   │
│                 │                       │                    │
│    ┌────────────▼────────────┐  ┌──────▼──────────────┐     │
│    │  FinancialDataExtractor │  │ QualitativeAnalysis │     │
│    │  Tool (via LLM)         │  │ Tool (RAG + LLM)    │     │
│    └────────────┬────────────┘  └──────┬──────────────┘     │
│                 │                      │                     │
│  ┌──────────────▼──────────────────────▼────────────┐        │
│  │         Document Ingestion Pipeline              │        │
│  │  - Download PDFs from Screener.in                │        │
│  │  - Parse using Marker or PyPDF2                  │        │
│  │  - Chunk & embed for vector DB                   │        │
│  └──────────────┬──────────────────────────────────┘         │
│                 │                                             │
│  ┌──────────────▼──────────────────────────────────┐         │
│  │  Data Sources & External APIs                    │         │
│  │  - Screener.in (PDFs)                            │         │
│  │  - Optional: Yahoo Finance (market data)         │         │
│  │  - Optional: Finnhub (live stock prices)         │         │
│  └───────────────────────────────────────────────────┘        │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                  MySQL 8.0 Logger (Async)                     │
│  - Logs all requests, full responses, errors                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

- **AI-First**: Every significant extraction or analysis step leverages LLMs
- **Modular Design**: Tools are independently testable and composable
- **Async Throughout**: FastAPI + asyncio for non-blocking I/O
- **Guardrails**: Prompt engineering, validation, and error recovery mechanisms
- **Transparency**: Complete audit trail for governance

---

# 2. System Architecture

## Component Breakdown

### 2.1 Document Acquisition Layer

**Responsibility**: Fetch and cache financial documents

**Key Capabilities**:
- Scrape or manually download PDFs from https://www.screener.in/company/TCS/consolidated/#documents
- Support for alternative sources (e.g., NSE, TCS investor relations)
- Caching mechanism to avoid redundant downloads
- Metadata extraction (date, quarter, source URL)

**Technology**:
- `requests` + `beautifulsoup4` for web scraping (or manual upload)
- `pathlib` for file management
- Optional: `selenium` for JavaScript-heavy sites

### 2.2 Document Parsing & Chunking

**Responsibility**: Convert PDFs to structured text and split for embedding

**Key Capabilities**:
- Multi-format PDF parsing (text-based and scanned images)
- Intelligent chunking respecting table boundaries and section headers
- Metadata tagging (source, page number, section type)
- Deduplication

**Technology**:
- `Marker` (advanced PDF parsing with layout understanding) OR
- `PyPDF2` + `pymupdf` (for text extraction)
- `LangChain.document_loaders.PyPDFLoader` for integration
- `LangChain.text_splitter.RecursiveCharacterTextSplitter`

### 2.3 Embedding & Vector Store

**Responsibility**: Create searchable semantic index of financial documents

**Key Capabilities**:
- Generate embeddings for document chunks
- Store and retrieve via similarity search
- Metadata filtering for financial domain (e.g., quarter, metric type)

**Technology**:
- **Embedding Model**: OpenAI `text-embedding-3-small` or Google `text-embedding-004`
- **Vector Store**: 
  - `Pinecone` (fully managed, enterprise-grade)
  - `Weaviate` (open source, on-prem option)
  - `Qdrant` (lightweight alternative)
- **Integration**: LangChain's `Retriever` interface

### 2.4 LLM & Reasoning Stack

**Responsibility**: Core intelligence for extraction, analysis, and forecast synthesis

**Key Capabilities**:
- Structured extraction from documents (via function calling)
- Semantic understanding of earnings call transcripts
- Multi-step reasoning and forecast synthesis

**Technology**:
- **Primary LLM**: OpenAI GPT-4 Turbo or Claude 3.5 Sonnet (for cost/capability trade-off)
- **Alternative**: Google Gemini 2.0 (emerging, competitive)
- **Framework**: LangChain for tool abstraction and chaining
- **Structured Output**: Pydantic models for guaranteed JSON schema adherence

### 2.5 Tool Implementations

#### 2.5.1 FinancialDataExtractorTool

Extracts key metrics from financial reports.

**Inputs**:
- PDF content or parsed text of financial statement

**Outputs** (Pydantic model):
```python
class FinancialMetrics(BaseModel):
    quarter: str  # e.g., "Q3 FY25"
    total_revenue: Optional[float]
    net_profit: Optional[float]
    operating_margin: Optional[float]
    ebitda: Optional[float]
    eps: Optional[float]
    yoy_growth: Optional[float]
    qoq_growth: Optional[float]
    source_document: str
    extraction_confidence: float  # 0-1
```

**Master Prompt**:
```
You are a financial analyst extracting key metrics from TCS quarterly reports.
Extract ONLY metrics that are explicitly stated in the document.
For each metric, provide:
1. The exact value with units
2. The period (quarter/year)
3. Source quote from the document
4. Confidence score (0-1) based on clarity of the statement

Return results as JSON matching this schema: [FinancialMetrics schema]
```

#### 2.5.2 QualitativeAnalysisTool

Performs RAG-based search and analysis of earnings call transcripts.

**Inputs**:
- Query describing the analysis (e.g., "What are management's concerns about client spending?")

**Outputs** (Pydantic model):
```python
class QualitativeInsight(BaseModel):
    theme: str
    management_sentiment: str  # "positive", "neutral", "negative"
    direct_quotes: List[str]
    recurring_mentions: int  # How many times theme mentioned across calls
    relevance_to_forecast: str  # Short explanation
    source_dates: List[str]
```

**Master Prompt**:
```
You are analyzing TCS earnings call transcripts to identify management outlook signals.
Using the retrieved transcript segments, identify:
1. Key themes mentioned by management
2. Sentiment (positive/neutral/negative) towards each theme
3. Direct quotes supporting the assessment
4. Frequency of mentions (indicates priority)
5. Relevance to next-quarter forecast

Return structured insights matching [QualitativeInsight schema]
```

#### 2.5.3 MarketDataTool (Optional)

Fetches live market context.

**Inputs**:
- None (fetches current data for TCS)

**Outputs** (Pydantic model):
```python
class MarketData(BaseModel):
    stock_price: float
    market_cap: float
    pe_ratio: float
    dividend_yield: float
    retrieved_at: str  # ISO timestamp
    source: str
```

**Integration Points**:
- Yahoo Finance API (free, lightweight)
- Finnhub API (premium, more reliable)

### 2.6 Agent Orchestrator

**LangChain ReAct Agent** with tool-calling capability.

**Reasoning Loop**:
1. **Parse**: Agent receives forecast request
2. **Think**: LLM decides which tool(s) to invoke based on task
3. **Act**: Execute tools in sequence or parallel
4. **Observe**: Receive tool results
5. **Synthesize**: Aggregate results into coherent forecast

**Tool Calling Pattern**:
```
Agent Input: "Generate Q4 FY25 forecast for TCS"
    ↓
Agent Thinks: "I need quantitative metrics + qualitative outlook"
    ↓
Agent Acts:
  - Tool 1: FinancialDataExtractor(latest_quarterly_report.pdf)
  - Tool 2: QualitativeAnalysis("Management outlook on spending trends")
  - Tool 3: MarketData() [optional]
    ↓
Agent Observes: 
  - Metrics: Revenue +5.2% YoY, margin stable
  - Sentiment: Cautious optimism, client caution
  - Price: $35.40, PE: 21
    ↓
Agent Synthesizes:
  "Next quarter forecast: Moderate growth (3-5%) with stable margins,
   tempered by client spending hesitancy identified in calls"
```

### 2.7 FastAPI Layer

**Endpoints**:

1. **POST /forecast**
   - Request: `{"company": "TCS"}`
   - Response: `ForecastResponse` (JSON as specified)
   - Async processing with request/response logging

2. **GET /health**
   - Liveness probe

**Request/Response Flow**:
```
Client ─(HTTP)→ FastAPI ─(async)→ Agent ─(tools)→ LLMs/APIs
                   ↓
                MySQL Logger ← all requests + responses
```

### 2.8 Logging & Audit

**MySQL Table**: `api_request_logs`

**Fields**:
- `id` (auto-increment, PK)
- `timestamp` (datetime)
- `request_payload` (JSON)
- `response_payload` (JSON)
- `status` (varchar: 'success' or 'error')
- `error_messages` (JSON array)
- `user_id` (nullable)

**Async Logging**:
- Non-blocking inserts using `aiomysql` or `asyncmy`
- Batch writes for high throughput
- Retry logic with exponential backoff

---

# 3. Tool & Agent Design

## 3.1 FinancialDataExtractorTool Design

### Purpose
Systematically extract financial metrics from TCS quarterly reports to quantify business performance.

### Detailed Master Prompt

```
You are an expert financial analyst specializing in extracting quantitative metrics 
from corporate financial statements. You are analyzing a TCS (Tata Consultancy Services) 
quarterly financial report.

**Your Task**:
Extract the following key financial metrics from the provided document:

1. **Revenue Metrics**:
   - Total Revenue (consolidated)
   - Revenue by segment (IT Services, Consulting, etc.)
   - YoY growth %
   - QoQ growth %

2. **Profitability Metrics**:
   - Net Profit / Net Income
   - Operating Margin
   - Net Profit Margin
   - EBITDA (if disclosed)

3. **Per-Share Metrics**:
   - Earnings Per Share (EPS)
   - Book Value Per Share (if available)

4. **Other KPIs**:
   - Operating Cash Flow
   - Return on Equity (if disclosed)
   - Employee Count
   - Attrition Rate (if disclosed)

**Extraction Guidelines**:
- Only extract metrics EXPLICITLY stated in the document.
- Do NOT estimate or infer values.
- For each metric:
  a) State the exact value with unit (INR Crore, USD Million, %)
  b) Specify the period (Quarter, Fiscal Year)
  c) Provide the direct quote from the document
  d) Assign confidence (0-1):
     - 1.0 = Explicitly stated in bold/tables
     - 0.8 = Clearly stated in body text
     - 0.6 = Stated but requires interpretation
     - 0.4 = Partially inferred from context
     - 0.0 = Not found / inferred (EXCLUDE)

**Output Format**:
Return a JSON array of extracted metrics:

[
  {
    "metric_name": "Total Revenue",
    "value": 62613,
    "unit": "INR Crore",
    "period": "Q1 FY25",
    "yoy_growth": 5.4,
    "yoy_growth_unit": "%",
    "source_quote": "Revenue of Rs 62,613 crore in Q1FY25, growth of 5.4% YoY",
    "page_reference": "3",
    "confidence": 1.0
  },
  ...
]

**Important**:
- If a metric is not found, do NOT include it (don't use null).
- Round percentages to 1 decimal; amounts to integers unless fractional.
- Always reference the exact quote to enable verification.
```

### Tool Implementation Pseudocode

```python
@tool
def financial_data_extractor(document_path: str) -> FinancialMetrics:
    """
    Extract key financial metrics from a TCS quarterly report PDF.
    
    Args:
        document_path: Path to PDF file
    
    Returns:
        FinancialMetrics object with extracted values
    """
    # Step 1: Load and parse PDF
    raw_text = load_pdf(document_path)
    
    # Step 2: Prepare document for LLM
    # - Split into chunks if >8000 tokens
    # - Prioritize tables and key sections
    
    # Step 3: Call LLM with master prompt
    extraction_prompt = FINANCIAL_EXTRACTION_MASTER_PROMPT + raw_text
    response = llm.invoke(extraction_prompt, temperature=0.0)
    
    # Step 4: Parse JSON response into Pydantic model
    metrics = FinancialMetrics.model_validate_json(response)
    
    # Step 5: Validate & Enrich
    # - Cross-check calculations (e.g., margin = profit / revenue)
    # - Flag missing critical metrics
    # - Add metadata (document source, extraction timestamp)
    
    return metrics
```

### Error Handling

1. **PDF Parse Failure**: Fallback to OCR (Tesseract) + retry
2. **LLM Hallucination**: Validate all extracted values exist in source text
3. **Malformed JSON**: Retry with stricter schema enforcement
4. **Missing Metrics**: Flag as warning, don't block forecast

### Validation Gates

```python
def validate_extraction(metrics: FinancialMetrics) -> ValidationResult:
    """Ensure extracted metrics are consistent."""
    
    checks = [
        # Revenue > 0
        lambda m: m.total_revenue > 0 if m.total_revenue else True,
        
        # Margin = Profit / Revenue (tolerance: ±2%)
        lambda m: abs(m.net_profit_margin - (m.net_profit / m.total_revenue))
                  < 0.02 if all([m.net_profit, m.total_revenue]) else True,
        
        # YoY growth reasonable (-50% to +50%)
        lambda m: -0.5 <= m.yoy_growth <= 0.5 if m.yoy_growth else True,
        
        # Confidence scores in valid range [0, 1]
        lambda m: 0 <= m.extraction_confidence <= 1,
    ]
    
    failed = [i for i, check in enumerate(checks) if not check(metrics)]
    
    return ValidationResult(
        is_valid=len(failed) == 0,
        failed_checks=failed,
        metrics=metrics
    )
```

---

## 3.2 QualitativeAnalysisTool Design

### Purpose
Analyze earnings call transcripts to identify management sentiment, forward guidance, and risks.

### RAG Architecture

```
Earnings Call Transcripts (2-3 recent calls)
    ↓
[Parser] → Extract Q&A sections, identify speakers (CFO, CEO, analysts)
    ↓
[Chunker] → Split into 500-token segments with metadata (timestamp, speaker)
    ↓
[Embedder] → Generate embeddings (e.g., OpenAI text-embedding-3-small)
    ↓
[Vector Store] → Store in Pinecone/Weaviate with metadata filters
    ↓
[Retriever] → On query, fetch top-5 relevant segments using similarity search
    ↓
[LLM Synthesis] → Analyze retrieved context + custom prompts
    ↓
QualitativeInsight output
```

### Master Prompt for Qualitative Analysis

```
You are analyzing TCS earnings call transcripts to extract forward-looking insights 
and management sentiment. Use ONLY the provided transcript segments.

**Analysis Objectives**:

1. **Management Outlook**:
   - What is management's tone about next quarter?
   - Are they guiding up, stable, or down?
   - Any forward guidance statements?

2. **Key Themes**:
   - What topics dominate the discussion?
   - Client spending trends?
   - Hiring/attrition outlook?
   - Geopolitical/macro risks?
   - Technology opportunities (AI, cloud)?

3. **Sentiment Analysis**:
   - Overall sentiment: Positive | Neutral | Negative
   - By theme (e.g., "positive on AI, cautious on client spending")

4. **Recurring Signals**:
   - Themes mentioned multiple times indicate management priorities
   - Note repetition across multiple calls

5. **Risk Disclosure**:
   - Explicitly mentioned risks
   - Implicit concerns (hesitancy in tone)

**Output Format**:

{
  "analysis_date": "2025-01-17",
  "themes": [
    {
      "theme": "Client Spending Caution",
      "sentiment": "negative",
      "quotes": [
        "We're seeing a pause in discretionary spending from our top clients",
        "BFSI clients remain cautious on investment commitments"
      ],
      "mentions_count": 3,
      "forecast_relevance": "High – suggests Q4 growth may moderate"
    },
    ...
  ],
  "overall_sentiment": "mixed",
  "guidance_statements": [...],
  "key_risks": [...],
  "confidence": 0.85
}

**Important**:
- ONLY use information from provided transcript segments.
- Direct quotes MUST match the source text exactly (use Ctrl+F to verify).
- If a theme has insufficient evidence, mark confidence < 0.7 and note it.
```

### Tool Implementation Pseudocode

```python
@tool
def qualitative_analysis_tool(analysis_query: str) -> QualitativeInsight:
    """
    Perform RAG-based analysis of TCS earnings call transcripts.
    
    Args:
        analysis_query: e.g., "What is management's outlook on client spending?"
    
    Returns:
        QualitativeInsight with themes, sentiment, quotes
    """
    
    # Step 1: Embed the query
    query_embedding = embedder.embed_text(analysis_query)
    
    # Step 2: Retrieve relevant transcript segments
    retrieved_docs = vector_store.similarity_search(
        query_embedding,
        k=5,
        filters={"document_type": "earnings_call"}
    )
    
    # Step 3: Prepare context for LLM
    context = "\n\n".join([
        f"[{doc.metadata['date']}] {doc.page_content}"
        for doc in retrieved_docs
    ])
    
    # Step 4: Generate analysis
    analysis_prompt = QUALITATIVE_MASTER_PROMPT + f"""
    
    Query: {analysis_query}
    
    Relevant Transcript Segments:
    {context}
    
    Provide your analysis:
    """
    
    response = llm.invoke(analysis_prompt, temperature=0.2)  # Lower temp for consistency
    
    # Step 5: Parse and validate
    insight = QualitativeInsight.model_validate_json(response)
    
    # Step 6: Quote Verification (important for auditability)
    for quote in insight.direct_quotes:
        if quote not in context:
            logger.warning(f"Quote not found in context: {quote}")
    
    return insight
```

### Pre-Processing: Preparing Transcripts for RAG

```python
def prepare_earnings_calls_for_rag():
    """
    Pipeline to ingest and chunk earnings call transcripts.
    """
    
    # Step 1: Download or load transcripts
    transcripts = download_from_screener()  # Manual or API
    
    for transcript_file in transcripts:
        
        # Step 2: Parse (likely HTML or plain text)
        raw_text = parse_transcript(transcript_file)
        
        # Step 3: Extract metadata
        metadata = extract_metadata(raw_text)
        # {date, quarter, ceo_name, cffo_name, analyst_count, ...}
        
        # Step 4: Segment by Q&A sections
        sections = segment_by_section(raw_text)
        # Separate opening remarks, Q&A, closing
        
        # Step 5: Create chunks with sliding window
        chunks = []
        for section in sections:
            for i in range(0, len(section), 400):  # 400-token overlap
                chunk_text = section[i:i+800]
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "section": section.type,
                        "position": i
                    }
                })
        
        # Step 6: Generate embeddings
        for chunk in chunks:
            embedding = embedder.embed_text(chunk["text"])
            chunk["embedding"] = embedding
        
        # Step 7: Upsert into vector store
        vector_store.upsert(chunks)
        
        logger.info(f"Ingested {len(chunks)} chunks from {transcript_file}")
```

---

## 3.3 Agent Chaining & Orchestration

### ReAct Agent Loop

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

AGENT_SYSTEM_PROMPT = """
You are a financial analysis expert generating quarterly business outlook forecasts 
for TCS (Tata Consultancy Services). You have access to specialized tools for 
extracting financial metrics and analyzing management sentiment from earnings calls.

**Your Process**:
1. Receive a forecast request (e.g., "Generate Q4 FY25 forecast")
2. Invoke tools to gather quantitative and qualitative data
3. Synthesize findings into a reasoned, structured forecast
4. Return forecast adhering to the required JSON schema

**Tools Available**:
- FinancialDataExtractor: Extract key metrics from latest quarterly report
- QualitativeAnalysis: Analyze earnings call transcripts for management outlook
- MarketData (optional): Fetch current stock price and market metrics

**Guidelines**:
- Use tools even if it seems you have the information; ensure current data
- Cross-validate insights (e.g., revenue growth + management sentiment should align)
- If tools return conflicting signals, flag the inconsistency
- Always cite sources (document names, quotes)
- Provide forecast with rationale for each key assumption

**Forecast Structure**:
- Summary outlook (1-2 sentences)
- Key financial trends (3-5 bullet points)
- Management's forward guidance (direct quotes)
- Risks and opportunities (2-3 each)
- Source documents (URLs/paths)
"""

# Initialize tools
tools = [
    financial_data_extractor_tool,
    qualitative_analysis_tool,
    # market_data_tool,  # Optional
]

# Create ReAct agent
agent = create_react_agent(
    llm=llm_model,
    tools=tools,
    prompt=PromptTemplate.from_template(AGENT_SYSTEM_PROMPT)
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

# Invoke for forecast
result = agent_executor.invoke({
    "input": "Generate a Q4 FY25 business outlook forecast for TCS"
})
```

### Error Recovery & Fallback

```python
class AgentWithFallback:
    """Wrapper adding resilience to agent execution."""
    
    def invoke_with_fallback(self, query: str, max_retries: int = 3) -> dict:
        """Execute agent with retry and fallback logic."""
        
        for attempt in range(max_retries):
            try:
                result = self.agent_executor.invoke({"input": query})
                return result
            
            except ToolExecutionError as e:
                logger.warning(f"Tool execution failed (attempt {attempt+1}): {e}")
                
                # Fallback: Try a simpler tool chain
                if attempt == max_retries - 1:
                    # Last attempt: return partial forecast
                    return self.generate_partial_forecast()
                
                # Retry with modified prompt
                modified_query = f"{query}\n\nPrevious error: {str(e)}\nTry alternative approach."
                continue
            
            except (TimeoutError, RateLimitError) as e:
                logger.error(f"External API error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # All retries exhausted
        return {
            "forecast_summary": "Unable to generate forecast; check logs",
            "errors": ["Max retries exceeded; see logs for details"]
        }
    
    def generate_partial_forecast(self) -> dict:
        """Return forecast with available data + error flags."""
        return {
            "forecast_summary": "Partial forecast (some data unavailable)",
            "key_financial_trends": [],
            "management_outlook": "Unable to retrieve",
            "risks_and_opportunities": [],
            "errors": ["Forecast generation incomplete"]
        }
```

---

# 4. AI Stack & Reasoning Logic

## 4.1 Complete AI Stack

| Component | Options | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **Primary LLM** | GPT-4 Turbo, Claude 3.5 Sonnet, Gemini 2.0 | Claude 3.5 Sonnet | Strong at financial reasoning, cost-effective, excellent structured output |
| **Embedding Model** | text-embedding-3-small, text-embedding-004, BAAI/bge-base | text-embedding-3-small | Fast, affordable, good quality for financial terms |
| **Vector Store** | Pinecone, Weaviate, Qdrant | Pinecone | Fully managed, enterprise reliability, SOC2 certified |
| **Document Parser** | Marker, PyPDF2, pdfplumber | Marker | Handles tables, images, preserves layout |
| **Framework** | LangChain, LlamaIndex, DSPy | LangChain | Best tool ecosystem, mature, active community |
| **Web Framework** | FastAPI, Flask, Django | FastAPI | Async-first, high performance, auto-docs |
| **Database (Logging)** | MySQL 8.0, PostgreSQL | MySQL 8.0 | Per requirements, sufficient for audit logging |
| **Async Driver** | aiomysql, asyncmy | asyncmy | Better performance, active maintenance |

## 4.2 Reasoning Approach

### Multi-Step Reasoning Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Forecast Request                                         │
│ "Generate Q4 FY25 outlook for TCS"                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Step 1: Information Gathering  │
        │ - Fetch latest financial data  │
        │ - Analyze management sentiment │
        │ - Retrieve market context      │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ Step 2: Quantitative Analysis  │
        │ - Calculate trend slopes       │
        │ - Identify anomalies           │
        │ - Project growth rates         │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ Step 3: Qualitative Analysis   │
        │ - Extract management guidance  │
        │ - Assess risk mentions         │
        │ - Gauge sentiment shifts       │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ Step 4: Synthesis & Reasoning  │
        │ - Cross-validate signals       │
        │ - Flag inconsistencies         │
        │ - Weight inputs (quant vs qual)│
        │ - Generate forecast narrative  │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ Step 5: Structure & Output     │
        │ - Format as required JSON      │
        │ - Cite sources                 │
        │ - Log to MySQL                 │
        └────────┬───────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────┐
        │ Output: ForecastResponse       │
        │ {forecast_summary, trends, ... │
        └────────────────────────────────┘
```

### Error & Ambiguity Handling

#### 1. **Hallucination Prevention**

```python
class HallucinationGuard:
    """Prevent LLM from inventing data."""
    
    def extract_with_verification(self, 
                                   document_text: str,
                                   extraction_prompt: str) -> dict:
        """
        Extract data, then verify all claims exist in source.
        """
        # Step 1: Extract
        extraction = self.llm.invoke(extraction_prompt)
        
        # Step 2: Verify each extracted claim
        for claim in extraction.get("claims", []):
            if claim["source_quote"] not in document_text:
                # Hallucination detected!
                logger.error(f"Hallucination: {claim['value']} not found")
                claim["verified"] = False
                claim["confidence"] = 0.0
            else:
                claim["verified"] = True
        
        return extraction
```

#### 2. **Conflicting Signals**

```python
class ConflictResolver:
    """Handle conflicting signals between quantitative and qualitative."""
    
    def resolve_conflict(self, quant_signal, qual_signal):
        """
        When quantitative and qualitative signals diverge, investigate.
        
        Example: Revenue up 5% (quant) but management sounds cautious (qual)
        → Flag as opportunity for deeper analysis
        """
        
        if quant_signal["direction"] != qual_signal["direction"]:
            # Investigate root cause
            investigation_prompt = f"""
            Conflicting signals detected:
            - Quantitative: {quant_signal['description']}
            - Qualitative: {qual_signal['description']}
            
            Possible explanations:
            1. One-time item vs. structural trend
            2. Timing lag (recent quarter vs. forward guidance)
            3. Segment divergence (some strong, some weak)
            
            Analyze and reconcile:
            """
            
            reconciliation = self.llm.invoke(investigation_prompt)
            
            return {
                "resolution": reconciliation,
                "confidence": 0.7,  # Lower due to conflict
                "flag": "USER_REVIEW_RECOMMENDED"
            }
```

#### 3. **Incomplete Data**

```python
class DataGapHandler:
    """Gracefully handle missing information."""
    
    def handle_missing_metric(self, metric_name: str, available_data: dict):
        """
        If critical metric unavailable, provide context about impact.
        """
        
        critical_metrics = ["revenue", "net_profit", "margin"]
        
        if metric_name in critical_metrics and metric_name not in available_data:
            logger.warning(f"Critical metric missing: {metric_name}")
            
            # Attempt to infer from related metrics
            inferred_value = self.infer_from_proxies(metric_name, available_data)
            
            if inferred_value:
                return {
                    "value": inferred_value,
                    "method": "inferred",
                    "confidence": 0.5,  # Low confidence
                    "note": "Based on proxy metrics; verify separately"
                }
            else:
                return {
                    "value": None,
                    "impact": "forecast confidence reduced",
                    "recommendation": "Attempt manual research"
                }
```

## 4.3 Prompt Engineering & Guardrails

### Temperature & Sampling Configuration

```python
# For extraction: Low temperature (deterministic)
extraction_config = {
    "temperature": 0.0,  # Greedy decoding
    "top_p": 1.0,
    "max_tokens": 2000,
}

# For synthesis: Moderate temperature (creative but grounded)
synthesis_config = {
    "temperature": 0.3,  # Low randomness
    "top_p": 0.95,
    "frequency_penalty": 0.5,  # Reduce repetition
}

# Structured output enforcement
output_schema = FinancialMetrics  # Pydantic model
llm_structured = llm.with_structured_output(output_schema)
```

### Chain-of-Thought Prompting

```python
COT_TEMPLATE = """
Answer the question step by step:

1. **Identify**: What information is needed?
2. **Source**: Where does this come from (which document/tool)?
3. **Extract**: What is the exact value from the source?
4. **Validate**: Does this value make sense given context?
5. **Conclude**: What does this mean for the forecast?

Question: {question}

Let's work through this step by step:
"""
```

### Confidence Scoring

```python
class ConfidenceScorer:
    """Assign confidence to LLM outputs."""
    
    def score_extraction(self, extraction: dict) -> float:
        """
        Confidence factors:
        - Is data explicitly stated? (+0.3)
        - Is value directly quoted? (+0.3)
        - Consistent across sources? (+0.2)
        - Aligned with historical trends? (+0.2)
        """
        
        score = 0.0
        
        if extraction.get("is_explicit"):
            score += 0.3
        
        if extraction.get("has_quote"):
            score += 0.3
        
        if extraction.get("cross_validated"):
            score += 0.2
        
        if extraction.get("fits_trend"):
            score += 0.2
        
        return min(score, 1.0)
```

---

# 5. Setup Instructions

## 5.1 Environment Preparation

### Prerequisites

```bash
# System Requirements
- Python 3.10+
- MySQL 8.0 (local or remote)
- 4GB RAM minimum
- Internet connectivity (for APIs)
```

### Step 1: Clone & Create Virtual Environment

```bash
# Clone repository (replace with actual repo)
git clone https://github.com/your-org/tcs-forecast-agent.git
cd tcs-forecast-agent

# Create virtual environment
python -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# requirements.txt should include:
fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.5
langchain-community==0.0.10
langchain-openai==0.0.5  # or langchain-anthropic
pydantic==2.5.0
pydantic-settings==2.1.0
requests==2.31.0
aiohttp==3.9.1
aiomysql==0.2.0  # or asyncmy
python-dotenv==1.0.0
marker-pdf==0.3.0
pymupdf==1.23.8
pinecone-client==3.0.0  # or weaviate-client, qdrant-client
openai==1.3.0  # or anthropic, google-generativeai
beautifulsoup4==4.12.2
lxml==4.9.3
pdfplumber==0.10.3
```

### Step 3: Configure Environment Variables

Create `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
# OR for Claude:
ANTHROPIC_API_KEY=sk-ant-...
# OR for Google:
GOOGLE_API_KEY=...

# Vector Store
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=tcs-financial-docs
PINECONE_ENVIRONMENT=gcp-starter

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=tcs_forecast

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Step 4: Set Up MySQL Database

```bash
# Connect to MySQL
mysql -h localhost -u root -p

# Execute initialization script
source database/init.sql

# Verify tables created
SHOW TABLES;
```

**init.sql contents**:
```sql
CREATE DATABASE IF NOT EXISTS tcs_forecast;
USE tcs_forecast;

CREATE TABLE api_request_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    request_payload JSON,
    response_payload JSON,
    status VARCHAR(20),
    error_messages JSON,
    user_id VARCHAR(255) NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE document_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_name VARCHAR(255) UNIQUE,
    document_url VARCHAR(1000),
    quarter VARCHAR(50),
    downloaded_at DATETIME,
    file_path VARCHAR(500),
    INDEX idx_quarter (quarter)
);
```

### Step 5: Initialize Vector Store

```python
# scripts/init_vector_store.py

from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if not exists
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="gcp",
            region="us-west1"
        )
    )
    print(f"Index {index_name} created")
else:
    print(f"Index {index_name} already exists")

# Run this once:
# python scripts/init_vector_store.py
```

---

## 5.2 Data Preparation

### Step 6: Download TCS Financial Documents

```python
# scripts/download_documents.py

import requests
from pathlib import Path

SCREENER_URL = "https://www.screener.in/company/TCS/consolidated/#documents"

def download_tcs_documents():
    """
    Download latest TCS quarterly reports from Screener.in
    (Manual approach recommended due to site structure)
    """
    
    documents = [
        {
            "name": "Q3_FY25_Results.pdf",
            "url": "https://www.screener.in/...",  # Replace with actual URL
            "quarter": "Q3 FY25"
        },
        # Add more documents
    ]
    
    download_dir = Path("data/financial_reports")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in documents:
        response = requests.get(doc["url"])
        file_path = download_dir / doc["name"]
        file_path.write_bytes(response.content)
        print(f"Downloaded {doc['name']}")
        
        # Log to database
        log_document_download(doc, str(file_path))

if __name__ == "__main__":
    download_tcs_documents()
```

### Step 7: Ingest & Vectorize Documents

```python
# scripts/ingest_documents.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings

def ingest_documents():
    """Load, chunk, embed, and store financial documents."""
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    loader = PyPDFLoader("data/financial_reports/Q3_FY25_Results.pdf")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(docs)
    
    # Add metadata
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["document_type"] = "financial_report"
    
    # Vectorize and store
    vectorstore = Pinecone.from_documents(
        splits,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print(f"Ingested {len(splits)} chunks into vector store")
    
    # Log ingestion
    log_ingestion_event(len(splits), "Q3 FY25")

if __name__ == "__main__":
    ingest_documents()
    
# Run: python scripts/ingest_documents.py
```

---

# 6. Running the Service

## 6.1 Start FastAPI Application

### Local Development

```bash
# Terminal 1: Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Output:
# Uvicorn running on http://0.0.0.0:8000
# Swagger UI available at http://0.0.0.0:8000/docs
```

### Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t tcs-forecast-agent .
docker run -p 8000:8000 --env-file .env tcs-forecast-agent
```

## 6.2 Test Endpoints

### Health Check

```bash
curl http://localhost:8000/health

# Response:
# {"status": "healthy", "timestamp": "2025-01-17T21:33:00Z"}
```

### Generate Forecast

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "company": "TCS"
  }'

# Response (see section 7 for full schema):
{
  "forecast_summary": "TCS expects moderate revenue growth (3-5%) in Q4 FY25, with stable margins despite macro caution from clients.",
  "key_financial_trends": [
    "Q3 Revenue: Rs 62,613 Cr, +5.4% YoY",
    "Operating Margin: Stable at ~21%",
    "Client Spending: Cautious but not declining"
  ],
  "management_outlook": "Management remains optimistic on long-term AI opportunity while acknowledging near-term client spending caution.",
  "risks_and_opportunities": [...],
  "source_documents": [...],
  "errors": []
}
```

---

# 7. Logging & Database Schema

## 7.1 API Request Logging

**What Gets Logged**:

1. **Request Payload**: Full incoming JSON
2. **Response Payload**: Complete forecast JSON
3. **Errors**: Any exceptions or validation failures
4. **Metadata**: Timestamp, user_id (if applicable), processing time

**Table: api_request_logs**

```sql
CREATE TABLE api_request_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    request_payload JSON,
    response_payload JSON,
    status VARCHAR(20),  -- 'success' or 'error'
    error_messages JSON,  -- Array of error strings
    user_id VARCHAR(255) NULL,
    processing_time_ms INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_timestamp (timestamp),
    INDEX idx_user_id (user_id)
);
```

**Example Log Entry**:

```json
{
  "id": 1,
  "timestamp": "2025-01-17 21:33:00",
  "request_payload": {
    "company": "TCS"
  },
  "response_payload": {
    "forecast_summary": "...",
    "key_financial_trends": [...],
    ...
  },
  "status": "success",
  "error_messages": [],
  "user_id": null,
  "processing_time_ms": 3200,
  "created_at": "2025-01-17 21:33:00"
}
```

## 7.2 Async Logging Implementation

```python
# app/services/logger.py

import aiomysql
from typing import Any, Dict
import json
from datetime import datetime

class AsyncDatabaseLogger:
    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool
    
    async def log_request(self,
                          request: Dict[str, Any],
                          response: Dict[str, Any],
                          status: str,
                          errors: list,
                          user_id: str = None,
                          processing_time_ms: int = None):
        """Non-blocking log to MySQL."""
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    query = """
                        INSERT INTO api_request_logs
                        (request_payload, response_payload, status, error_messages, user_id, processing_time_ms)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    
                    await cur.execute(query, (
                        json.dumps(request),
                        json.dumps(response),
                        status,
                        json.dumps(errors),
                        user_id,
                        processing_time_ms
                    ))
                    
                    await conn.commit()
        
        except Exception as e:
            # Logging failure should not crash the app
            logger.error(f"Failed to log request: {e}")
```

## 7.3 Error Handling & Policies

| Error Type | Handling | Log Level |
|------------|----------|-----------|
| **Document Not Found** | Return 404 + error message | WARNING |
| **LLM Rate Limit** | Retry with exponential backoff (up to 3x) | WARNING |
| **MySQL Connection Failure** | Queue log, attempt retry later | ERROR |
| **Malformed Input** | Return 422 validation error | WARNING |
| **Tool Execution Timeout** | Return partial forecast + timeout flag | ERROR |
| **Hallucination Detected** | Exclude metric, flag in response | WARNING |

---

# 8. Evaluation & Guardrails

## 8.1 Prompt Evaluation Framework

### Metric 1: Source Fidelity

**Objective**: Ensure extracted data comes from source documents

```python
def evaluate_source_fidelity(extraction: dict, source_text: str) -> float:
    """
    Score: 0-1
    - 1.0: All claims verifiable in source
    - 0.5: Some claims lack direct quotes
    - 0.0: Claims not in source (hallucination)
    """
    
    verified_count = 0
    total_count = len(extraction.get("metrics", []))
    
    for metric in extraction.get("metrics", []):
        quote = metric.get("source_quote", "")
        if quote and quote in source_text:
            verified_count += 1
    
    return verified_count / total_count if total_count > 0 else 0.0
```

### Metric 2: Consistency Check

**Objective**: Ensure extracted values are mathematically consistent

```python
def evaluate_consistency(metrics: dict) -> float:
    """
    Score: 0-1
    Check relationships:
    - Revenue = all segments sum
    - Net Profit Margin = Net Profit / Revenue
    - Etc.
    """
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Segment sum
    if "segments" in metrics:
        checks_total += 1
        segment_sum = sum(s["revenue"] for s in metrics["segments"])
        if abs(segment_sum - metrics["total_revenue"]) < 100:
            checks_passed += 1
    
    # Check 2: Margin calculation
    if "net_profit" in metrics and "total_revenue" in metrics:
        checks_total += 1
        calculated_margin = metrics["net_profit"] / metrics["total_revenue"]
        if abs(calculated_margin - metrics.get("net_margin", 0)) < 0.02:
            checks_passed += 1
    
    return checks_passed / checks_total if checks_total > 0 else 1.0
```

### Metric 3: Forecast Accuracy (Post-Deployment)

**Objective**: Compare forecasted metrics vs. actuals in next period

```python
def evaluate_forecast_accuracy(forecast: dict, actuals: dict) -> dict:
    """
    Compare forecast to actual results once available.
    - Revenue forecast error %
    - Direction correctness (up/down/flat)
    - Confidence calibration
    """
    
    errors = {}
    
    for metric in ["revenue_growth", "margin", "eps"]:
        forecast_val = forecast.get(metric)
        actual_val = actuals.get(metric)
        
        if forecast_val and actual_val:
            error_pct = abs(forecast_val - actual_val) / actual_val * 100
            errors[metric] = error_pct
    
    # Aggregate
    mape = sum(errors.values()) / len(errors) if errors else None
    
    return {
        "mape": mape,
        "errors": errors,
        "acceptable": mape < 15 if mape else None  # <15% acceptable
    }
```

## 8.2 Retry & Recovery Strategies

### Exponential Backoff for Transient Failures

```python
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """Execute function with exponential backoff on failure."""
    
    for attempt in range(max_retries):
        try:
            return await func()
        
        except (asyncio.TimeoutError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            
            delay = base_delay * (backoff_factor ** attempt)
            logger.warning(f"Attempt {attempt+1} failed; retrying in {delay}s")
            await asyncio.sleep(delay)
```

### Graceful Degradation

```python
async def invoke_agent_with_degradation(query: str) -> dict:
    """
    Attempt full forecast, fall back to partial on failure.
    """
    
    try:
        # Attempt full forecast
        return await agent_executor.ainvoke({"input": query})
    
    except ToolExecutionError:
        logger.warning("Tool execution failed; attempting reduced scope")
        
        # Fallback: Try with fewer tools
        try:
            return await simplified_agent.ainvoke({"input": query})
        except:
            logger.error("Simplified agent also failed")
            
            # Final fallback: Return cached forecast if available
            cached = get_cached_forecast()
            if cached:
                return {
                    **cached,
                    "warning": "Using cached forecast; live analysis unavailable"
                }
            else:
                raise ForecastUnavailableError()
```

## 8.3 Quality Assurance

### Pre-Deployment Validation Checklist

```python
class QAValidator:
    """Comprehensive validation before marking forecast as ready."""
    
    def validate_forecast(self, forecast: dict) -> ValidationResult:
        checks = [
            self.check_schema_compliance,
            self.check_source_citations,
            self.check_no_hallucinations,
            self.check_forecast_logic,
            self.check_confidence_scores,
            self.check_error_handling,
        ]
        
        results = []
        for check in checks:
            try:
                passed, details = check(forecast)
                results.append({
                    "check": check.__name__,
                    "passed": passed,
                    "details": details
                })
            except Exception as e:
                results.append({
                    "check": check.__name__,
                    "error": str(e)
                })
        
        all_passed = all(r.get("passed", False) for r in results)
        
        return ValidationResult(
            passed=all_passed,
            checks=results,
            recommend_human_review=not all_passed
        )
    
    def check_schema_compliance(self, forecast: dict) -> tuple:
        """Ensure forecast adheres to required JSON schema."""
        required_keys = [
            "forecast_summary", "key_financial_trends",
            "management_outlook", "risks_and_opportunities",
            "source_documents", "errors"
        ]
        
        missing = [k for k in required_keys if k not in forecast]
        
        return len(missing) == 0, {"missing_keys": missing}
    
    def check_source_citations(self, forecast: dict) -> tuple:
        """Verify all claims are cited."""
        
        # Placeholder: Detailed implementation would parse forecast
        # and check each claim against source documents
        
        return True, {"citations_checked": True}
    
    # ... more checks
```

---

# 9. Discussion of Trade-Offs

## 9.1 Limitations Identified

### Limitation 1: Document Availability

**Problem**: TCS financial documents on Screener.in may be incomplete or delayed

**Impact**: 
- Forecast for next quarter may have incomplete prior quarter data
- Earnings transcripts may not be available immediately

**Mitigation**:
- Implement document cache with fallback to multiple sources (NSE, TCS IR, BSE)
- Accept partial forecasts with data availability flags
- Manual upload capability as fallback

### Limitation 2: LLM Hallucination in Financial Domain

**Problem**: 
- LLMs can confabulate financial figures when trained data conflicts
- Financial document formats vary widely, confusing LLM parsing

**Impact**:
- Extracted metrics may be incorrect, leading to faulty forecasts
- User might act on false data

**Mitigation**:
- Implement source verification (quoted text must exist in document)
- Use low temperature (0.0) for extraction to reduce variation
- Cross-validate metrics using calculation rules (e.g., margin = profit/revenue)
- Flag low-confidence extractions
- Require human review before critical decisions

### Limitation 3: Earnings Call Transcript Quality

**Problem**:
- Transcripts may be auto-generated (transcription errors)
- Incomplete or redacted portions
- Timing lag before transcript availability

**Impact**:
- Qualitative analysis may miss nuance or contain errors
- Forecast may lack forward guidance

**Mitigation**:
- Prioritize official company-published transcripts over auto-generated
- Accept transcription-error tolerance in quote verification
- Use multiple calls (not just latest) for stronger signals
- Manual review of critical management quotes

### Limitation 4: Market Context Lag

**Problem**:
- Stock prices, peer metrics update in real-time but historical data has lag
- Forecast may reference stale market context

**Impact**:
- Market data snapshot in forecast may not reflect current conditions

**Mitigation**:
- Include timestamp for all fetched market data
- Document data freshness in forecast response
- Optional: Real-time data subscription (higher cost)

### Limitation 5: Forecast Accuracy Bounded by Model Capability

**Problem**:
- LLMs cannot predict future unknown events (black swans)
- Forecast is inherently uncertain

**Impact**:
- Forecast may diverge significantly from actuals
- User might overestimate confidence

**Mitigation**:
- Always frame forecast as "outlook" based on current data, not prediction
- Provide confidence ranges and sensitivity analysis
- Document key assumptions
- Update forecast with new quarterly data

## 9.2 Design Trade-Offs

| Trade-Off | Option A | Option B | Chosen | Rationale |
|-----------|----------|----------|--------|-----------|
| **LLM Provider** | GPT-4 (expensive, best quality) | Claude 3.5 (balanced) | Claude 3.5 | Better cost-quality for financial reasoning |
| **Vector Store** | Pinecone (managed) | Weaviate (open-source) | Pinecone | Enterprise reliability > cost savings |
| **Embedding Model** | text-embedding-3-large (better) | text-embedding-3-small (faster) | small | Speed acceptable, cost savings significant |
| **Error Recovery** | Strict (fail fast) | Lenient (fallbacks) | Lenient | Better user experience, audit trail handles failures |
| **Logging** | MySQL (per requirement) | Postgres (better JSON) | MySQL | Meets requirements, sufficient for audit |
| **Document Format** | PDF-only | Accept multiple formats | PDF-only | Financial reports predominantly PDF |
| **Forecast Frequency** | On-demand | Scheduled batch | On-demand | Flexibility preferred, real-time data needs |

## 9.3 Mitigation Strategies Summary

| Limitation | Primary Mitigation | Secondary Mitigation | Monitoring |
|-----------|-------------------|----------------------|------------|
| Document lag | Multi-source fetch | Manual upload | Track document freshness |
| Hallucination | Source verification | Confidence scores | Audit log review |
| Transcript quality | Official transcripts first | Multiple-call analysis | Quote accuracy spot-checks |
| Market lag | Timestamp all data | Accept as limitation | Include freshness in response |
| Forecast accuracy | Confidence ranges | Sensitivity analysis | Compare to actuals post-quarter |

---

# 10. Appendices

## 10.1 Complete Code Structure

```
tcs-forecast-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Settings & environment
│   ├── models/
│   │   ├── __init__.py
│   │   ├── forecast.py         # Pydantic models for request/response
│   │   └── financial.py        # Financial metric models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent.py            # LangChain agent orchestrator
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── financial_extractor.py
│   │   │   ├── qualitative_analysis.py
│   │   │   └── market_data.py
│   │   ├── document_processor.py  # PDF parsing, chunking
│   │   ├── vector_store.py        # Pinecone integration
│   │   └── logger.py              # MySQL logging
│   └── routes/
│       ├── __init__.py
│       ├── forecast.py         # /forecast endpoint
│       └── health.py           # /health endpoint
├── scripts/
│   ├── init_vector_store.py
│   ├── download_documents.py
│   ├── ingest_documents.py
│   └── test_agent.py
├── database/
│   └── init.sql
├── tests/
│   ├── test_extraction.py
│   ├── test_agent.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── Dockerfile
└── docker-compose.yml
```

## 10.2 Example: Complete FastAPI Endpoint

```python
# app/routes/forecast.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import time
import logging

router = APIRouter(prefix="/api", tags=["forecast"])
logger = logging.getLogger(__name__)

class ForecastRequest(BaseModel):
    company: str
    include_market_data: bool = False

@router.post("/forecast", response_model=dict)
async def generate_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    agent_service = Depends(get_agent_service),
    logger_service = Depends(get_logger_service)
):
    """
    Generate business outlook forecast for specified company.
    
    Args:
        request: ForecastRequest with company name
        background_tasks: FastAPI background task runner
        agent_service: Injected agent orchestrator
        logger_service: Injected async logger
    
    Returns:
        ForecastResponse JSON
    """
    
    start_time = time.time()
    
    try:
        # Validate company
        if request.company.upper() != "TCS":
            raise HTTPException(
                status_code=400,
                detail="Currently only TCS is supported"
            )
        
        logger.info(f"Forecast request received for {request.company}")
        
        # Invoke agent
        forecast_result = await agent_service.generate_forecast(
            company=request.company,
            include_market_data=request.include_market_data
        )
        
        # Structure response
        response = {
            "forecast_summary": forecast_result["summary"],
            "key_financial_trends": forecast_result["trends"],
            "management_outlook": forecast_result["outlook"],
            "risks_and_opportunities": forecast_result["risks"],
            "source_documents": forecast_result["sources"],
            "errors": forecast_result.get("errors", [])
        }
        
        if request.include_market_data:
            response["market_data"] = forecast_result.get("market_data")
        
        # Log asynchronously
        processing_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(
            logger_service.log_request,
            request=request.dict(),
            response=response,
            status="success",
            errors=[],
            processing_time_ms=processing_time_ms
        )
        
        return JSONResponse(status_code=200, content=response)
    
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        
        error_response = {
            "forecast_summary": "",
            "key_financial_trends": [],
            "management_outlook": "",
            "risks_and_opportunities": [],
            "source_documents": [],
            "errors": [str(e)]
        }
        
        # Log error asynchronously
        processing_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(
            logger_service.log_request,
            request=request.dict(),
            response=error_response,
            status="error",
            errors=[str(e)],
            processing_time_ms=processing_time_ms
        )
        
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
```

## 10.3 Example: Agent Initialization

```python
# app/services/agent.py

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.services.tools import (
    financial_data_extractor,
    qualitative_analysis,
    market_data_tool
)

class AgentService:
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=config.openai_api_key
        )
        self.tools = [
            financial_data_extractor,
            qualitative_analysis,
            # market_data_tool  # Optional
        ]
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Initialize ReAct agent."""
        
        system_prompt = """
        You are a financial analyst generating quarterly business outlooks for TCS.
        Use the available tools to gather data, then synthesize into a forecast.
        ... (as detailed in section 3.3)
        """
        
        prompt = PromptTemplate.from_template(system_prompt)
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        return executor
    
    async def generate_forecast(self, company: str, **kwargs) -> dict:
        """Generate forecast for given company."""
        
        input_query = f"Generate Q4 FY25 business outlook forecast for {company}"
        
        try:
            result = await self.agent.ainvoke({"input": input_query})
            return self._parse_agent_output(result)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
    
    def _parse_agent_output(self, agent_result: dict) -> dict:
        """Parse agent output into forecast structure."""
        # Implementation details...
        pass
```

## 10.4 Unit Test Example

```python
# tests/test_extraction.py

import pytest
from app.services.tools import financial_data_extractor
from app.models.financial import FinancialMetrics

@pytest.fixture
def sample_pdf_path():
    return "tests/fixtures/Q3_FY25_Sample.pdf"

@pytest.mark.asyncio
async def test_financial_extraction(sample_pdf_path):
    """Test financial data extraction."""
    
    result = await financial_data_extractor(sample_pdf_path)
    
    # Assertions
    assert isinstance(result, FinancialMetrics)
    assert result.total_revenue > 0
    assert 0 <= result.operating_margin <= 1
    assert 0 <= result.extraction_confidence <= 1
    
    # Verify source quotes
    assert len(result.source_quote) > 0

@pytest.mark.asyncio
async def test_extraction_confidence_scores(sample_pdf_path):
    """Test that confidence scores reflect data clarity."""
    
    result = await financial_data_extractor(sample_pdf_path)
    
    # Explicitly stated metrics should have high confidence
    if result.total_revenue:
        assert result.extraction_confidence >= 0.8

@pytest.mark.asyncio
async def test_extraction_validation(sample_pdf_path):
    """Test validation of extracted metrics."""
    
    result = await financial_data_extractor(sample_pdf_path)
    
    # Validate mathematical consistency
    if result.net_profit and result.total_revenue:
        calculated_margin = result.net_profit / result.total_revenue
        assert abs(calculated_margin - result.net_profit_margin) < 0.02
```

## 10.5 Error Scenario Examples

### Scenario 1: PDF Download Failure

```
User Request:
  POST /forecast {"company": "TCS"}

System Behavior:
  1. Agent tries to fetch latest financial report
  2. Download fails (server unavailable)
  3. Catch: ToolExecutionError
  4. Retry: Exponential backoff (1s, 2s, 4s)
  5. All retries exhausted
  6. Fallback: Check cached documents
  7. Found: Q3 FY25 from 2 weeks ago
  8. Proceed with historical data
  9. Return forecast with warning:
     {
       "forecast_summary": "...",
       "errors": ["Latest Q4 data unavailable; using Q3 FY25 data"]
     }

Logging:
  INSERT api_request_logs
  status: "success" (partial)
  error_messages: ["Latest Q4 data unavailable..."]
```

### Scenario 2: Hallucinated Metric

```
Extraction Process:
  1. LLM processes financial report PDF
  2. Extracts: "Total Revenue: Rs 150,000 Cr (Q4 FY25)"
  3. Verification check: Search for "150,000" in source PDF
  4. Result: NOT FOUND (hallucination!)
  5. LLM actually said Rs 62,000 Cr in Q3
  6. Mark as hallucination

Response:
  {
    "key_financial_trends": [
      "Q3 Revenue: Rs 62,613 Cr (VERIFIED)",
      "Q4 FY25: Data unavailable (hallucination detected in extraction)"
    ],
    "errors": ["Revenue extraction failed verification; excluded from forecast"]
  }

Logging:
  error_messages: ["Hallucination detected: Revenue claim unverified"]
```

### Scenario 3: Vector Store Unavailable

```
User Request:
  POST /forecast (qualitative analysis requested)

System Behavior:
  1. Agent invokes QualitativeAnalysisTool
  2. Tool attempts to connect to Pinecone
  3. ConnectionError: Unable to reach vector store
  4. Catch: ToolExecutionError with retries (exponential backoff)
  5. Max retries exceeded
  6. Graceful degradation: Fallback to quantitative-only forecast
  7. Return forecast with warning:
     {
       "forecast_summary": "Growth outlook based on quantitative data only",
       "management_outlook": "Unable to retrieve (vector store unavailable)",
       "errors": ["Qualitative analysis unavailable; see logs"]
     }

Logging:
  status: "success" (partial)
  processing_time_ms: 5200
  error_messages: ["Vector store unavailable after 3 retries"]
```

---

## README.md Template

See comprehensive README template below:

```markdown
# TCS Business Outlook Forecast Agent

An AI-powered FastAPI service that generates quarterly business forecasts for 
Tata Consultancy Services (TCS) by analyzing financial reports and earnings calls.

## Quick Start

### 1. Clone & Setup
\`\`\`bash
git clone https://github.com/your-org/tcs-forecast-agent.git
cd tcs-forecast-agent
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
\`\`\`

### 2. Configure Environment
\`\`\`bash
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, PINECONE_API_KEY, MySQL credentials)
\`\`\`

### 3. Initialize Database & Vector Store
\`\`\`bash
mysql -u root -p < database/init.sql
python scripts/init_vector_store.py
python scripts/ingest_documents.py
\`\`\`

### 4. Run Service
\`\`\`bash
uvicorn app.main:app --reload
# Open http://localhost:8000/docs for interactive API documentation
\`\`\`

### 5. Generate Forecast
\`\`\`bash
curl -X POST http://localhost:8000/forecast \\
  -H "Content-Type: application/json" \\
  -d '{"company": "TCS", "include_market_data": true}'
\`\`\`

## AI Stack

### LLM & Reasoning
- **Primary LLM**: Claude 3.5 Sonnet (reasoning, extraction)
- **Embedding**: OpenAI text-embedding-3-small (1536 dims)
- **Framework**: LangChain (agent orchestration, tool calling)

### Data Infrastructure
- **Vector Store**: Pinecone (enterprise RAG)
- **Document Parser**: Marker (advanced PDF → structured text)
- **Async DB**: asyncmy (MySQL 8.0 logging)

### Guardrails & Safety
- **Source Verification**: All extracted metrics verified against source text
- **Hallucination Detection**: Claims must match document quotes
- **Confidence Scoring**: Each extraction includes 0-1 confidence
- **Schema Enforcement**: Structured output via Pydantic models

## Architecture

```
[FastAPI] → [LangChain Agent] → [Financial Data Extractor Tool]
                             ↘  [Qualitative Analysis Tool (RAG)]
                             ↘  [Market Data Tool (Optional)]
                             ↓
                        [MySQL Logging]
```

## Tools & Capabilities

### FinancialDataExtractorTool
Extracts key metrics from TCS quarterly financial reports:
- Revenue, net profit, margins
- Segment-wise performance
- YoY/QoQ growth rates
- EPS, ROE, cash flow

**Validation**: All extracted values verified in source document with confidence scoring.

### QualitativeAnalysisTool
RAG-based analysis of earnings call transcripts:
- Management sentiment analysis
- Key themes and concerns
- Forward guidance extraction
- Risk mentions and opportunities

**Process**: 
1. Parse recent earnings call transcripts
2. Chunk and embed for vector search
3. Retrieve relevant segments for query
4. Analyze retrieved context with LLM
5. Aggregate insights across multiple calls

### MarketDataTool (Optional)
Fetches live market context:
- Stock price, market cap, PE ratio
- Dividend yield
- Peer comparison metrics

## API Endpoints

### POST /forecast
Generates quarterly business outlook forecast.

**Request**:
```json
{
  "company": "TCS",
  "include_market_data": true
}
```

**Response**:
```json
{
  "forecast_summary": "TCS expects moderate growth...",
  "key_financial_trends": ["Revenue +5.4% YoY", "Margin stable"],
  "management_outlook": "Cautious optimism on AI...",
  "risks_and_opportunities": [
    "Risk: Client spending caution",
    "Opportunity: AI services demand"
  ],
  "source_documents": ["Q3 FY25 Results PDF", "Earnings Call Transcript"],
  "errors": []
}
```

### GET /health
Health check endpoint.

## Database Schema

**api_request_logs**:
- `id` (PK, auto-increment)
- `timestamp` (request received time)
- `request_payload` (JSON)
- `response_payload` (JSON)
- `status` ('success' or 'error')
- `error_messages` (JSON array)
- `user_id` (optional)

## Error Handling

The system implements graceful degradation:
- **Tool Failure**: Retries with exponential backoff (1s, 2s, 4s)
- **Partial Data**: Returns forecast with available data + error flags
- **Fallback**: Uses cached data if live sources unavailable

All errors logged to MySQL with full request/response context.

## Development

### Running Tests
\`\`\`bash
pytest tests/ -v
pytest tests/test_extraction.py -k "test_confidence"
\`\`\`

### Adding New Documents
\`\`\`bash
# Download new reports and place in data/financial_reports/
python scripts/ingest_documents.py
\`\`\`

### Monitoring
- Check logs: `tail -f app.log`
- View forecast quality: Query `api_request_logs` for status/errors
- Monitor vector store: Pinecone dashboard

## Limitations & Trade-Offs

- **Document Lag**: Forecasts delayed if latest reports not yet available
- **LLM Hallucination**: Mitigated via source verification; user review recommended
- **Accuracy**: Inherently uncertain; view as data-driven outlook, not prediction
- **Transcript Quality**: Depends on availability of official transcripts

See [LIMITATIONS.md](LIMITATIONS.md) for detailed discussion.

## License

MIT

## Support

Report issues: https://github.com/your-org/tcs-forecast-agent/issues
```

---

## Conclusion

This comprehensive guide provides:

1. **Clear architecture** with component breakdown
2. **Detailed tool design** with master prompts and pseudocode
3. **Step-by-step setup** from environment to first forecast
4. **Production-grade patterns** for logging, error handling, and guardrails
5. **Real-world examples** and error scenarios
6. **Honest discussion** of limitations and trade-offs

The system balances **capability** (multi-source analysis, AI reasoning) with **reliability** (guardrails, error recovery) and **transparency** (audit logging, source citation).

**Next Steps**:
1. Customize master prompts for your specific analysis needs
2. Test with Q3 FY25 sample documents
3. Deploy MySQL and Pinecone (or alternatives)
4. Run end-to-end tests and validate forecast quality
5. Monitor post-deployment and iterate on guardrails

Good luck with your implementation!
```

