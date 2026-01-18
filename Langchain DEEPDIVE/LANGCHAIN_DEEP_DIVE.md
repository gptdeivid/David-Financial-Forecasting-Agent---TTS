# 🔗 LangChain Deep Dive: Complete Knowledge Guide for TCS Forecast Agent

**Target**: AI engineers, cloud architects, ML engineers
**Complexity**: Intermediate to Advanced
**Prerequisites**: Python 3.10+, FastAPI basics, LLM API familiarity

---

## TABLE OF CONTENTS

1. [LangChain Fundamentals](#1-langchain-fundamentals)
2. [Architecture & Core Components](#2-architecture--core-components)
3. [Agent Framework: The ReAct Pattern](#3-agent-framework-the-react-pattern)
4. [Tool Design & Integration](#4-tool-design--integration)
5. [RAG Systems & Vector Stores](#5-rag-systems--vector-stores)
6. [LangChain vs LangGraph: Decision Framework](#6-langchain-vs-langgraph-decision-framework)
7. [Agent Orchestration & Chaining](#7-agent-orchestration--chaining)
8. [Memory & State Management](#8-memory--state-management)
9. [Error Handling & Robustness](#9-error-handling--robustness)
10. [TCS Forecast Agent: LangChain Implementation](#10-tcs-forecast-agent-langchain-implementation)

---

## 1. LangChain Fundamentals

### What is LangChain?

LangChain is a **framework for developing applications powered by large language models (LLMs)**. It enables:

- **LLM Integration**: Connect to OpenAI, Claude, Google, Anthropic, and open-source models
- **Agent Orchestration**: Build agents that reason, plan, and execute actions
- **RAG Pipelines**: Combine LLMs with external data sources for grounded responses
- **Memory Management**: Maintain conversation context and state across interactions
- **Tool Integration**: Connect to 600+ external services, databases, and APIs

### Why LangChain for TCS Forecast Agent?

| Need | LangChain Solution |
|------|-------------------|
| Multi-tool financial analysis | Tool framework with built-in chaining |
| Document analysis (earnings calls, financial reports) | RAG + vector store integrations |
| Structured agent reasoning | ReAct pattern implementation |
| Error resilience & retries | Built-in error handling strategies |
| State persistence | Memory backends (Redis, databases) |
| FastAPI integration | LCEL (LangChain Expression Language) |

### LangChain's Design Philosophy

```
┌─────────────────────────────────────────────────────────┐
│  LangChain's "Composable" Design Philosophy             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. MODULARITY                                          │
│     Each component (LLM, tool, memory) is standalone   │
│     Swap components without rewriting core logic        │
│                                                         │
│  2. COMPOSABILITY                                       │
│     Chain components using | (pipe) operator           │
│     component_a | component_b | component_c            │
│                                                         │
│  3. FLEXIBILITY                                         │
│     Use high-level abstractions or low-level APIs      │
│     Choose between ease-of-use and fine-grained control│
│                                                         │
│  4. INTEGRATIONS                                        │
│     600+ built-in connectors to external services      │
│     Minimal boilerplate for common patterns             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Architecture & Core Components

### The LangChain Stack

```
┌────────────────────────────────────────────────────┐
│  APPLICATION LAYER (FastAPI, Flask, etc.)         │
│  ├─ API Endpoints                                  │
│  ├─ Request/Response handling                      │
│  └─ Business logic orchestration                   │
├────────────────────────────────────────────────────┤
│  LANGGRAPH LAYER (Execution Engine) ⭐ MODERN     │
│  ├─ State machines with checkpointing             │
│  ├─ Runnable interface for composability          │
│  ├─ Support for human-in-the-loop                 │
│  └─ Built-in persistence & retries                │
├────────────────────────────────────────────────────┤
│  LANGCHAIN CORE LAYER (Framework)                 │
│  ├─ Agents (ReAct, OpenAIFunctions, etc.)        │
│  ├─ Chains (Sequential, mapping, etc.)            │
│  ├─ Memory (Buffer, Summary, Entity)              │
│  ├─ Tools (Registry, execution, validation)       │
│  ├─ Retrievers (Vector, BM25, Ensemble)           │
│  └─ Output parsers (Structured, JSON, etc.)       │
├────────────────────────────────────────────────────┤
│  LANGCHAIN COMMUNITY LAYER (Integrations)         │
│  ├─ LLM providers (OpenAI, Claude, etc.)          │
│  ├─ Vector stores (Pinecone, Weaviate, etc.)      │
│  ├─ Memory backends (Redis, PostgreSQL, etc.)     │
│  ├─ Document loaders (PDF, Web, Notion, etc.)     │
│  └─ Retrieval tools (Web search, Wikipedia, etc.) │
├────────────────────────────────────────────────────┤
│  EXTERNAL SERVICES                                 │
│  ├─ LLM APIs (GPT-4, Claude, Gemini, etc.)        │
│  ├─ Vector databases (Pinecone, Weaviate, Qdrant) │
│  ├─ Data sources (Screener.in, Yahoo Finance)     │
│  └─ Memory stores (Redis, PostgreSQL, MySQL)      │
└────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **LLMs (Language Models)**

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

# For TCS agent: Claude 3.5 Sonnet (financial reasoning)
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.0,  # Deterministic for extraction
    max_tokens=2048
)
```

**LLM Parameters for Financial Analysis**:
- **temperature**: 0.0 (extraction) → 0.4 (synthesis) to control randomness
- **max_tokens**: Depends on response complexity (2048+ for detailed analysis)
- **top_p**: 0.9-1.0 for financial data to avoid unlikely completions

#### 2. **Tools**

Tools are functions the agent can call to gather data or perform actions.

```python
from langchain_core.tools import tool

@tool
def extract_financial_metrics(pdf_path: str) -> dict:
    """Extract revenue, profit, margins from financial report.
    
    Args:
        pdf_path: Path to PDF financial report
        
    Returns:
        Dictionary with extracted metrics
    """
    # Implementation: PDF parsing + LLM extraction
    pass

@tool
def search_earnings_calls(company: str, quarter: str) -> str:
    """Search for earnings call transcripts.
    
    Args:
        company: Company name (e.g., "TCS")
        quarter: Quarter identifier (e.g., "Q3 FY25")
        
    Returns:
        Transcript text or URL
    """
    pass
```

#### 3. **Prompts & Templates**

```python
from langchain_core.prompts import ChatPromptTemplate

financial_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst specializing in metric extraction.
    
Your task: Extract key financial metrics from the provided text.
Return ONLY valid JSON with no markdown formatting.

Metrics to extract:
- Total Revenue
- Net Profit
- Operating Margin (%)
- EBITDA
- EPS (Earnings Per Share)

Constraints:
1. Use exact numbers from source
2. Include currency and units
3. Mark confidence (0.0-1.0) for each metric
4. Include source quote for each metric
"""),
    ("user", "{input}")
])
```

#### 4. **Runnable Interface (LCEL)**

LangChain Expression Language is a composable interface:

```python
from langchain_core.runnables import RunnableSequence

# Compose: extract metrics → parse JSON → validate
chain = (
    {"input": lambda x: x}
    | financial_extraction_prompt
    | llm
    | output_parser
    | {"metrics": lambda x: x, "confidence": calculate_confidence}
)

# Execute
result = chain.invoke({"input": financial_report_text})
```

#### 5. **Memory**

Maintain conversation state across agent steps:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    human_prefix="Analyst",
    ai_prefix="Agent"
)

# Agent automatically includes memory in each step
```

---

## 3. Agent Framework: The ReAct Pattern

### Understanding ReAct

**ReAct = Reason + Act + Observe**

ReAct is an agent pattern where the LLM:
1. **Reasons**: Analyzes the problem and decides which tools to use
2. **Acts**: Calls the appropriate tool with parameters
3. **Observes**: Receives tool output and incorporates it into reasoning
4. **Loops**: Repeats until it can provide a final answer

### ReAct Loop Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  START: "Generate Q4 FY25 forecast for TCS"                │
│    ↓                                                        │
│  [REASON] Agent thinks: "I need financial metrics"         │
│    ↓                                                        │
│  [ACT] Calls: extract_financial_metrics("Q3_FY25.pdf")    │
│    ↓                                                        │
│  [OBSERVE] Gets: {"revenue": "₹60,000Cr", ...}            │
│    ↓                                                        │
│  [REASON] Agent thinks: "Now I need qualitative signals"   │
│    ↓                                                        │
│  [ACT] Calls: analyze_earnings_call("earnings_call.txt")  │
│    ↓                                                        │
│  [OBSERVE] Gets: {"management_outlook": "positive", ...}   │
│    ↓                                                        │
│  [REASON] Agent thinks: "I can now synthesize forecast"    │
│    ↓                                                        │
│  [ACT] Calls: synthesize_forecast({metrics, sentiment})    │
│    ↓                                                        │
│  [OBSERVE] Gets: {"forecast": "Growth of 8-10%", ...}      │
│    ↓                                                        │
│  END: Returns final forecast JSON                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### ReAct Implementation (LangGraph Modern Approach)

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools
@tool
def financial_data_extractor(report_text: str) -> dict:
    """Extract financial metrics from report."""
    # Implementation
    pass

@tool
def qualitative_analyzer(transcript_text: str) -> dict:
    """Analyze earnings call for sentiment and themes."""
    # Implementation
    pass

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Create agent (LangGraph handles ReAct loop internally)
tools = [financial_data_extractor, qualitative_analyzer]
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a financial forecasting agent..."
)

# Execute
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Generate TCS Q4 FY25 forecast"
    }]
})
```

### Key Concepts in ReAct Loop

#### Intermediate Steps

The agent maintains a scratchpad of all Reason → Act → Observe cycles:

```python
{
    "intermediate_steps": [
        {
            "action": "extract_financial_metrics",
            "action_input": {"pdf_path": "q3_fy25.pdf"},
            "observation": {"revenue": 60000, "profit": 12000, ...}
        },
        {
            "action": "analyze_earnings_call",
            "action_input": {"transcript": "..."},
            "observation": {"sentiment": "positive", "themes": [...]}
        }
    ]
}
```

The agent sees this entire history when making next decision.

#### Agent Scratchpad

The scratchpad is a string representation of thought process:

```
Thought: I need to analyze TCS financials for Q4 forecast
Action: extract_financial_metrics
Action Input: {"pdf_path": "q3_fy25.pdf"}
Observation: {"revenue": 60000, "profit": 12000, "margin": 0.20}

Thought: Now I need sentiment from management
Action: analyze_earnings_call
Action Input: {"transcript": "earnings_q3.txt"}
Observation: {"sentiment": "positive", "guidance": "growth"}

Thought: I can now generate the forecast
Action: synthesize_forecast
Action Input: {"metrics": {...}, "sentiment": {...}}
Observation: {"forecast": "8-10% growth expected"}

Thought: I have enough information. Final Answer is ready.
```

---

## 4. Tool Design & Integration

### Tool Anatomy

```python
from langchain_core.tools import tool, ToolException
from typing import Annotated

@tool
def financial_data_extractor(
    report_text: Annotated[str, "The financial report text to analyze"],
    metrics: Annotated[list[str], "List of metrics to extract (e.g., ['revenue', 'profit'])"]
) -> dict:
    """Extract key financial metrics from a report.
    
    This tool uses LLM-powered extraction with confidence scoring.
    Returns metrics only if confidence > 0.6.
    
    Args:
        report_text: Full text of the financial report
        metrics: Which metrics to extract
        
    Returns:
        Dictionary with extracted metrics and confidence scores
        
    Raises:
        ToolException: If report text is invalid or too short
    """
    if not report_text or len(report_text) < 100:
        raise ToolException("Report text too short or empty")
    
    try:
        # LLM extraction with temperature=0.0
        extracted = _llm_extract_metrics(report_text, metrics)
        
        # Validate confidence
        validated = {
            k: v for k, v in extracted.items()
            if v.get("confidence", 0) >= 0.6
        }
        
        return validated
    except Exception as e:
        raise ToolException(f"Extraction failed: {str(e)}")
```

### Tool Best Practices for TCS Agent

#### 1. **Clear Descriptions**

```python
@tool
def extract_financial_metrics(pdf_path: str) -> dict:
    """Extract financial metrics from TCS quarterly report.
    
    This tool:
    - Parses PDF financial statements
    - Extracts: Revenue, Net Profit, Operating Margin, EBITDA, EPS
    - Validates against source documents
    - Returns confidence scores for each metric
    - Includes exact quotes from source
    
    Use this when you need current financial data for TCS.
    
    Args:
        pdf_path: Path to TCS Q3 or Q4 financial report
        
    Returns:
        dict with keys:
        - metrics: {metric_name: value}
        - sources: {metric_name: source_quote}
        - confidence: {metric_name: 0.0-1.0 score}
    """
```

#### 2. **Input Validation**

```python
@tool
def analyze_earnings_call(
    transcript_path: Annotated[str, "Path to earnings call transcript"],
    analysis_type: Annotated[str, "Type: 'sentiment', 'themes', 'guidance'"]
) -> dict:
    """Analyze earnings call for forward-looking statements.
    
    Raises ToolException for invalid inputs.
    """
    valid_types = ["sentiment", "themes", "guidance"]
    if analysis_type not in valid_types:
        raise ToolException(
            f"analysis_type must be one of {valid_types}, got {analysis_type}"
        )
    
    if not Path(transcript_path).exists():
        raise ToolException(f"File not found: {transcript_path}")
    
    # Proceed with analysis
    return _analyze_call(transcript_path, analysis_type)
```

#### 3. **Error Handling in Tools**

```python
@tool
def fetch_market_data(symbol: str, metric: str) -> dict:
    """Fetch current market data for TCS stock."""
    try:
        data = _fetch_from_api(symbol, metric)
        return {
            "value": data.value,
            "currency": data.currency,
            "timestamp": data.timestamp,
            "source": "Yahoo Finance"
        }
    except APITimeoutError as e:
        raise ToolException(f"API timeout: {str(e)} - Please retry")
    except ValueError as e:
        raise ToolException(f"Invalid metric '{metric}': {str(e)}")
    except Exception as e:
        raise ToolException(f"Unexpected error: {str(e)}")
```

### Tool Chaining Strategy for TCS Agent

Tools should work sequentially with output feeding into next:

```
Tool 1: Extract Financial Metrics
├─ Input: Q3 FY25 financial report
└─ Output: {"revenue": 60000, "profit": 12000, ...}
    │
    ├─ Tool 2A: Analyze Earnings Call
    │   └─ Output: {"sentiment": "positive", "guidance": [...]}
    │
    └─ Tool 2B: Fetch Market Data (parallel)
        └─ Output: {"stock_price": 3800, "pe_ratio": 28.5, ...}
            │
            └─ Tool 3: Synthesize Forecast
                ├─ Inputs: metrics + sentiment + market data
                └─ Output: {"forecast": "8-10% Q4 growth", ...}
```

---

## 5. RAG Systems & Vector Stores

### RAG Fundamentals

**RAG = Retrieval-Augmented Generation**

Instead of relying solely on LLM training data, RAG:
1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with retrieved context
3. **Generates** answer grounded in retrieved documents

### RAG Architecture for TCS Agent

```
┌────────────────────────────────────────────────────┐
│           RAG Pipeline for TCS Analysis            │
├────────────────────────────────────────────────────┤
│                                                    │
│  INPUT: "What did management say about growth?"  │
│    ↓                                              │
│  [1] RETRIEVAL                                    │
│  Query → Embed → Vector Search → Top-K Chunks   │
│  From: TCS earnings calls Q3, Q2 transcripts     │
│    ↓                                              │
│  [2] CONTEXT AUGMENTATION                        │
│  Retrieved: Management quotes about growth       │
│  Mgmt: "Growth is expected in IT services..."    │
│    ↓                                              │
│  [3] GENERATION                                  │
│  LLM sees: Original question + retrieved context│
│  LLM generates: Answer grounded in retrieved text│
│    ↓                                              │
│  OUTPUT: "According to management, growth is..." │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Setting Up RAG with LangChain

#### Step 1: Load and Split Documents

```python
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load TCS financial documents
loader = PDFPlumberLoader("tcs_q3_fy25_financial_report.pdf")
documents = loader.load()

# Split into chunks (semantic units)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Characters per chunk
    chunk_overlap=200,  # Overlap for context
    separators=["\n\n", "\n", ". ", " ", ""]  # Split by these, in order
)

chunks = text_splitter.split_documents(documents)

# Each chunk: {"page_content": "...", "metadata": {"source": "...", "page": 1}}
```

#### Step 2: Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import AnthropicEmbeddings

# Choose embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Fast, cheap, 1536-dim
    api_key="..."
)

# Each document chunk becomes a vector (1536 numbers)
# Example: "TCS revenue grew to 60,000 crores"
# → [0.234, -0.123, 0.456, ..., -0.789]  # 1536 dimensions
```

#### Step 3: Store in Vector Database

```python
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_weaviate import WeaviateVectorStore

# Option 1: Pinecone (managed, enterprise)
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="tcs-forecast-index",
    namespace="earnings-calls"
)

# Option 2: Chroma (local, lightweight)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### Step 4: Retrieve Relevant Context

```python
# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 5}  # Return top-5 chunks
)

# Retrieve for a query
query = "What did management say about growth drivers?"
retrieved_chunks = retriever.get_relevant_documents(query)

for chunk in retrieved_chunks:
    print(f"Content: {chunk.page_content[:200]}...")
    print(f"Source: {chunk.metadata['source']}, Page: {chunk.metadata['page']}")
```

#### Step 5: Integrate with LLM

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create RAG chain
prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Using only the context above, answer the question.
If not answerable from context, say "Not found in documents".
""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("What are growth drivers mentioned by management?")
```

### Vector Store Comparison for TCS Agent

| Store | Setup | Scalability | Cost | Best For |
|-------|-------|-------------|------|----------|
| **Pinecone** | Managed SaaS | 1B+ vectors | $1-5K/mo | Enterprise, high volume |
| **Weaviate** | Self-hosted | 100M+ vectors | Free (self) | Flexibility, hybrid search |
| **Qdrant** | Self-hosted | High | Free (self) | Performance, filtering |
| **Chroma** | Local/embedded | <1M vectors | Free | Development, small datasets |
| **Milvus** | Kubernetes | 1B+ vectors | Free (self) | Kubernetes deployments |

**Recommendation for TCS Agent**: 
- Development: Chroma (local, quick setup)
- Production: Pinecone (managed, reliable) or Weaviate (hybrid search for earnings calls)

### Hybrid Search (Keyword + Semantic)

For financial analysis, combine keyword search (exact matches) with semantic search:

```python
from langchain_retriever import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Semantic retriever (vector similarity)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Keyword retriever (BM25 - TF-IDF variant)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Combine with weights
ensemble = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)

# Hybrid search benefits:
# - Semantic: Finds "company growth momentum" for query "expansion"
# - Keyword: Finds exact phrase "EBITDA margin" for financial metric search
```

---

## 6. LangChain vs LangGraph: Decision Framework

### Historical Context

**LangChain (Legacy)**:
- AgentExecutor pattern
- Agent runs loop internally
- Less control over flow
- Limited state persistence

**LangGraph (Modern - Recommended 2025)**:
- Graph-based state machine
- Full control via nodes and edges
- Built-in checkpointing and persistence
- Better for production systems

### Decision Tree

```
Do you need to build an agent?
│
├─ Simple, single-step logic?
│  └─ Use LangChain Chains (LCEL)
│
├─ Multi-step reasoning with tools?
│  └─ Use LangGraph (ReAct agent)
│
├─ Complex workflow with branches?
│  ├─ Conditional routing?
│  ├─ Human-in-the-loop?
│  ├─ Long-running persistence?
│  └─ → Use LangGraph
│
└─ Enterprise production system?
   └─ Use LangGraph with:
      ├─ Persistent checkpoints
      ├─ Error recovery
      ├─ Monitoring/logging
      └─ Human oversight
```

### TCS Forecast Agent: Why LangGraph?

Your agent needs:
- ✅ Multi-tool orchestration (extraction + analysis + synthesis)
- ✅ Sequential reasoning (Reason → Act → Observe loops)
- ✅ Error resilience (tool failures shouldn't crash agent)
- ✅ State persistence (for audit/debugging)
- ✅ Async execution (FastAPI integration)

**Decision: Use LangGraph**

---

## 7. Agent Orchestration & Chaining

### Single-Agent Architecture (Your Case)

```
┌─────────────────────────────┐
│   User Request (FastAPI)    │
│  "Generate TCS forecast"    │
└──────────────┬──────────────┘
               │
        ┌──────▼──────┐
        │  ReAct Agent│ (LangGraph)
        ├──────────────┤
        │ Reasoning:   │
        │ • Analyze req│
        │ • Plan steps │
        │ • Decide tool│
        └──────────────┘
        ▲     │     │
        │     │     └─────────────────┐
        │     │                       │
   ┌────┴─────┴────────────────┐    │
   │   Tool Execution Loop     │    │
   ├──────────────────────────┤    │
   │ 1. Extract Metrics       │    │
   │ 2. Analyze Sentiment     │    │
   │ 3. Synthesize Forecast   │    │
   └──────────────────────────┘    │
        │                           │
        │  Tool Results             │
        │  {"revenue": 60000, ...}  │
        └───────────────────────────┘
               │
        ┌──────▼──────┐
        │   Output    │
        │  Forecast   │
        │   JSON      │
        └──────────────┘
```

### Multi-Agent Architecture (Optional Future)

If you need different agents for different domains:

```
┌──────────────────┐
│ Supervisor Agent │  (Main orchestrator)
└────────┬────┬────┘
         │    │
    ┌────▼┐  ┌▼────┐
    │Agent│  │Agent│  (Specialized)
    │  1  │  │  2  │
    │  FX │  │ ESG │
    └─────┘  └─────┘
```

**For TCS**: Single agent sufficient (all tasks are financial analysis).

### LCEL (LangChain Expression Language) for Chaining

LCEL allows composable, type-safe chaining:

```python
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# Define components
prompt_template = ChatPromptTemplate.from_template("...")
llm = ChatOpenAI(model="gpt-4-turbo")
output_parser = JsonOutputParser()

# Chain them with | (pipe) operator
chain = (
    {"input": RunnablePassthrough()}  # Pass input through
    | prompt_template  # Format prompt
    | llm  # Call LLM
    | output_parser  # Parse output
)

# Invoke
result = chain.invoke("input_value")

# LCEL Benefits:
# - Type checking: Each component validates input/output types
# - Streaming: Supports streaming outputs
# - Async: All components support async/.invoke()
# - Debugging: Clear error messages at each step
```

---

## 8. Memory & State Management

### Memory Types

#### 1. **Buffer Memory** (Store all messages)

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    human_prefix="User",
    ai_prefix="Assistant"
)

# For short conversations (< 10 turns)
# ✅ Accurate
# ❌ Grows unbounded, expensive
```

#### 2. **Summary Memory** (Compress history)

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    memory_key="chat_history",
    return_messages=True
)

# LLM periodically summarizes conversation
# ✅ Bounded size
# ❌ Loses details
```

#### 3. **Entity Memory** (Extract key facts)

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    entity_cache={
        "TCS": "Tata Consultancy Services - Indian IT services company",
        "FY25": "Financial year 2024-2025"
    }
)

# Maintains entity knowledge base
# ✅ Structured, searchable
# ❌ Limited context
```

### For TCS Agent: LangGraph State

LangGraph uses explicit state management (better than implicit memory):

```python
from typing import Annotated
from langgraph.graph import GraphState

class ForecastState(TypedDict):
    """State maintained across agent steps."""
    
    # Input/Output
    user_query: str
    forecast_result: dict
    
    # Intermediate data
    financial_metrics: dict
    qualitative_signals: dict
    market_data: dict
    
    # Metadata
    errors: list[str]
    tool_calls: list[dict]
    reasoning_trace: list[str]

# Agent can access and modify state in each step
def reasoning_node(state: ForecastState) -> ForecastState:
    # Decide which tool to use
    state["reasoning_trace"].append("Deciding tool...")
    return state
```

### Persistence & Checkpointing

Store agent state for recovery and audit:

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpoints
memory_saver = MemorySaver()

# Or PostgreSQL for production
from langgraph.checkpoint.postgres import PostgresSaver

postgres_saver = PostgresSaver(
    conn_string="postgresql://user:pass@localhost/langraph_db"
)

# LangGraph automatically saves state after each step
agent = graph.compile(checkpointer=postgres_saver)

# Recover from interrupt
config = {"configurable": {"thread_id": "forecast_001"}}
result = agent.invoke(input, config=config)

# Later, resume from same point
result = agent.invoke(
    {**input, "resume": True},
    config=config
)
```

---

## 9. Error Handling & Robustness

### Tool Execution Errors

```python
from langchain_core.tools import ToolException

@tool
def extract_metrics(pdf_path: str) -> dict:
    """Extract metrics with error handling."""
    try:
        if not Path(pdf_path).exists():
            raise ToolException(f"File not found: {pdf_path}")
        
        # LLM extraction might fail
        try:
            metrics = llm_extract(pdf_path)
        except APIConnectionError:
            raise ToolException("LLM API unavailable - retry later")
        except TokenLimitError:
            raise ToolException("Document too long for LLM")
        
        return metrics
        
    except ToolException:
        raise  # Re-raise ToolException
    except Exception as e:
        raise ToolException(f"Unexpected error: {str(e)}")
```

### Agent-Level Error Handling

```python
from langgraph.errors import NodeError

def create_agent_with_error_handling():
    # Each tool has max_retries
    tools_with_retry = [
        tool1.with_retry(max_retries=3, backoff_factor=2),
        tool2.with_retry(max_retries=3, backoff_factor=2)
    ]
    
    agent = create_react_agent(
        llm=llm,
        tools=tools_with_retry,
        max_iterations=15  # Prevent infinite loops
    )
    
    return agent

# In FastAPI endpoint
try:
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}],
    })
except NodeError as e:
    # Agent execution failed
    logger.error(f"Agent error: {e}")
    return {"error": "Analysis failed", "details": str(e)}
```

### Graceful Degradation

If some tools fail, continue with others:

```python
@tool
def fetch_market_data(symbol: str) -> dict:
    """Fetch market data with fallback."""
    try:
        return fetch_from_yahoo(symbol)
    except Exception as e:
        logger.warning(f"Market data fetch failed: {e}")
        # Return empty dict, agent continues without market data
        return {"error": "market_data_unavailable"}

# In agent logic:
def synthesis_node(state):
    metrics = state.get("financial_metrics", {})
    market = state.get("market_data", {})
    
    if "error" in market:
        logger.info("Proceeding without market data")
        # Continue with just financial metrics
    
    forecast = synthesize(metrics, market)
    return {"forecast_result": forecast}
```

### Retry Strategies

```python
import backoff

@backoff.on_exception(
    backoff.expo,  # Exponential backoff
    APIConnectionError,
    max_tries=3,
    base=2,
    factor=1
)
def extract_metrics_with_backoff(pdf_path):
    return llm_extract(pdf_path)

# Backoff schedule:
# Attempt 1: immediate
# Attempt 2: 2^1 = 2 seconds
# Attempt 3: 2^2 = 4 seconds
```

---

## 10. TCS Forecast Agent: LangChain Implementation

### Complete Architecture

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# ===== 1. LLM Configuration =====
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,  # Deterministic for extraction
    max_tokens=2048
)

# ===== 2. Tool 1: Financial Data Extractor =====
@tool
def financial_data_extractor(report_text: str) -> dict:
    """Extract financial metrics from TCS report.
    
    Extracts:
    - Total Revenue
    - Net Profit
    - Operating Margin
    - EBITDA
    - EPS
    
    Returns metrics with confidence scores and source quotes.
    """
    extraction_prompt = ChatPromptTemplate.from_template("""
    Extract financial metrics from this report:
    
    {report_text}
    
    Return JSON:
    {{
        "metrics": {{"revenue": {{...}}, "profit": {{...}}}},
        "confidence": {{"revenue": 0.95, ...}},
        "sources": {{"revenue": "quote from report", ...}}
    }}
    """)
    
    chain = extraction_prompt | llm | JsonOutputParser()
    result = chain.invoke({"report_text": report_text})
    
    # Filter low-confidence metrics
    result["metrics"] = {
        k: v for k, v in result["metrics"].items()
        if result["confidence"].get(k, 0) >= 0.6
    }
    
    return result

# ===== 3. Tool 2: Qualitative Analysis (RAG) =====
# Initialize vector store from earnings calls
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="tcs-earnings-calls",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def qualitative_analyzer(query: str) -> dict:
    """Analyze earnings calls for management sentiment.
    
    Searches earnings call transcripts for:
    - Management outlook
    - Growth drivers
    - Risk factors
    - Forward guidance
    """
    # Retrieve relevant context
    context_chunks = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([
        f"[{chunk.metadata.get('source')}]\n{chunk.page_content}"
        for chunk in context_chunks
    ])
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    Context from earnings calls:
    {context}
    
    Analyze for: {query}
    
    Return JSON:
    {{
        "sentiment": "positive/neutral/negative",
        "key_themes": ["theme1", "theme2"],
        "management_quotes": ["quote1", "quote2"],
        "risks": ["risk1", "risk2"],
        "guidance": "forward-looking statement"
    }}
    """)
    
    chain = analysis_prompt | llm | JsonOutputParser()
    return chain.invoke({
        "context": context_text,
        "query": query
    })

# ===== 4. Tool 3: Market Data Fetcher =====
@tool
def market_data_fetcher(symbol: str) -> dict:
    """Fetch current TCS market data."""
    try:
        data = yf.Ticker(symbol).info
        return {
            "stock_price": data.get("currentPrice"),
            "pe_ratio": data.get("trailingPE"),
            "market_cap": data.get("marketCap"),
            "retrieved_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Market data unavailable: {e}"}

# ===== 5. Create Agent =====
tools = [
    financial_data_extractor,
    qualitative_analyzer,
    market_data_fetcher
]

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    state_modifier="""You are a financial forecasting agent specializing in TCS (Tata Consultancy Services).

Your task: Generate a business outlook forecast for TCS's next quarter.

Process:
1. Extract latest financial metrics (revenue, profit, margins)
2. Analyze management's forward-looking statements from earnings calls
3. Consider current market conditions
4. Synthesize into a reasoned forecast

Focus on:
- Quantitative trends (3-quarter history)
- Qualitative signals (management guidance, risks)
- Market validation

Constraints:
- All claims must reference source documents
- Include confidence scores
- Highlight key risks and uncertainties
- Return structured JSON forecast"""
)

# ===== 6. FastAPI Integration =====
@app.post("/forecast")
async def generate_forecast(request: ForecastRequest):
    """Generate TCS business forecast."""
    
    config = {"configurable": {"thread_id": str(uuid4())}}
    
    try:
        # Invoke agent
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Generate TCS forecast for {request.quarter}"
            }]
        }, config=config)
        
        # Extract forecast from agent messages
        forecast_json = parse_forecast(result["messages"][-1].content)
        
        # Log to MySQL
        await log_to_database(
            request_payload=request.dict(),
            response_payload=forecast_json,
            status="success"
        )
        
        return ForecastResponse(**forecast_json)
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        await log_to_database(
            request_payload=request.dict(),
            response_payload={"error": str(e)},
            status="error"
        )
        return {"error": str(e)}
```

### Key Implementation Points

#### Temperature Settings for Each Tool

```python
# Tool 1: Extraction (Temperature = 0.0)
# Deterministic, exact metric extraction
llm_extract = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.0  # Greedy decoding
)

# Tool 2: Analysis (Temperature = 0.2)
# Consistent interpretation with minor variance
llm_analysis = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.2  # Low randomness
)

# Tool 3: Synthesis (Temperature = 0.4)
# Balanced creativity and grounding
llm_synthesis = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.4  # Moderate creativity
)
```

#### Confidence Scoring Integration

```python
def extract_with_confidence(text: str) -> dict:
    """Extract metrics with confidence scores."""
    
    extraction_prompt = """
    Extract metrics and for EACH metric provide:
    - value: extracted value
    - confidence: 0.0-1.0 based on:
        * Clear source quote (0.9+)
        * Interpreted from context (0.7-0.8)
        * Estimated/calculated (0.5-0.6)
    - source_quote: exact text from document
    """
    
    result = llm.invoke(extraction_prompt)
    
    # Filter by confidence threshold
    confident_metrics = {
        k: v for k, v in result.items()
        if result["confidence"].get(k, 0) >= 0.6
    }
    
    return confident_metrics
```

#### State Persistence for Audit

```python
# LangGraph automatically saves state
config = {"configurable": {"thread_id": "forecast_tcs_20250117"}}

result = agent.invoke(input_data, config=config)

# Can retrieve state later for audit
state_history = agent.get_state(config)

# Entire agent execution is traceable:
# - All tool calls
# - LLM reasoning
# - Intermediate results
# - Final forecast
```

---

## Summary: LangChain Knowledge for TCS Agent

### You Need to Understand:

1. ✅ **ReAct Agent Pattern**: How LLM reasons, acts, observes in loops
2. ✅ **Tool Design**: Creating extractors, analyzers, retrievers
3. ✅ **RAG Systems**: Vector stores for earnings call analysis
4. ✅ **LCEL**: Composing LLM, prompts, parsers
5. ✅ **LangGraph**: Modern agent framework with persistence
6. ✅ **Error Handling**: Graceful degradation and retries
7. ✅ **State Management**: Tracking agent execution for audit

### Implementation Path:

1. Start with basic agent setup (tools + LLM)
2. Add RAG for earnings call analysis
3. Implement error handling and retries
4. Add state persistence (LangGraph checkpointing)
5. Integrate with FastAPI and MySQL logging
6. Test and iterate on prompts

### Performance Targets:

- **Latency**: < 30 seconds per forecast
- **Tool execution**: < 20 seconds total
- **Error rate**: < 1% of requests
- **Hallucination rate**: < 5% (source verification guardrails)

---

**This guide serves as your LangChain reference. Use it alongside the starter code and TCS_Forecast_Agent_Guide.md for complete understanding.**
