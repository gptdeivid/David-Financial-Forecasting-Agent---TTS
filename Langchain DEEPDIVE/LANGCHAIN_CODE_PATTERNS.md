# 🚀 LangChain Practical Code Patterns for TCS Agent

**Purpose**: Ready-to-use code snippets and patterns specific to your TCS Forecast Agent
**Length**: ~4,000 words of implementation examples
**Format**: Copy-paste ready with explanations

---

## TABLE OF CONTENTS

1. [Basic Tool Implementation](#1-basic-tool-implementation)
2. [RAG Setup (Earnings Call Analysis)](#2-rag-setup-earnings-call-analysis)
3. [Agent with LangGraph](#3-agent-with-langgraph)
4. [Error Handling Patterns](#4-error-handling-patterns)
5. [FastAPI Integration](#5-fastapi-integration)
6. [Testing & Validation](#6-testing--validation)

---

## 1. Basic Tool Implementation

### Pattern 1A: Simple Financial Extractor

```python
from langchain_core.tools import tool
from langchain_core.tools import ToolException
from langchain_anthropic import ChatAnthropic
import json

@tool
def extract_financial_metrics(report_text: str) -> dict:
    """
    Extract financial metrics from TCS financial report.
    
    Extracts: Revenue, Net Profit, Operating Margin, EBITDA, EPS
    Returns: Metrics with confidence scores and source quotes
    """
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.0  # Deterministic extraction
    )
    
    extraction_prompt = f"""
    You are a financial analyst. Extract metrics from this report.
    
    REPORT TEXT:
    {report_text}
    
    EXTRACTION TASK:
    Extract these metrics if present:
    1. Total Revenue (annual)
    2. Net Profit (annual)
    3. Operating Margin (%)
    4. EBITDA (annual)
    5. EPS (Earnings Per Share)
    
    For EACH metric, provide:
    - "value": The extracted number with units
    - "confidence": 0.0-1.0 (0.9+ if exact quote, 0.7-0.8 if interpreted, 0.5-0.6 if calculated)
    - "source_quote": The exact text from report
    - "notes": Any caveats or context
    
    Return ONLY valid JSON, no markdown:
    {{
        "revenue": {{"value": "...", "confidence": 0.95, "source_quote": "...", "notes": "..."}},
        "net_profit": {{"value": "...", ...}},
        ...
    }}
    """
    
    try:
        # Call LLM
        response = llm.invoke(extraction_prompt)
        
        # Parse JSON response
        result = json.loads(response.content)
        
        # Validate confidence levels
        validated = {}
        for metric, data in result.items():
            if data.get("confidence", 0) >= 0.6:  # Only high-confidence metrics
                validated[metric] = data
        
        return {"metrics": validated, "count": len(validated)}
        
    except json.JSONDecodeError as e:
        raise ToolException(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise ToolException(f"Extraction failed: {str(e)}")
```

### Pattern 1B: Tool with Input Validation

```python
from pathlib import Path
from langchain_core.tools import tool
from typing import Annotated

@tool
def extract_from_pdf(
    pdf_path: Annotated[str, "Path to PDF file"],
    extraction_type: Annotated[str, "Type: 'metrics' or 'qualitative'"]
) -> dict:
    """
    Extract data from TCS PDF report.
    
    Args:
        pdf_path: Full path to PDF (e.g., '/data/tcs_q3_fy25.pdf')
        extraction_type: 'metrics' for financials, 'qualitative' for text
    """
    
    # Validation: File exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise ToolException(f"File not found: {pdf_path}")
    
    if pdf_file.suffix.lower() != ".pdf":
        raise ToolException(f"Expected PDF, got: {pdf_file.suffix}")
    
    # Validation: Extraction type
    valid_types = ["metrics", "qualitative"]
    if extraction_type not in valid_types:
        raise ToolException(
            f"extraction_type must be {valid_types}, got: {extraction_type}"
        )
    
    # Validation: File size (< 50MB)
    file_size_mb = pdf_file.stat().st_size / 1024 / 1024
    if file_size_mb > 50:
        raise ToolException(f"PDF too large: {file_size_mb:.1f}MB (max 50MB)")
    
    # Extract based on type
    try:
        if extraction_type == "metrics":
            return extract_financial_metrics_internal(pdf_path)
        else:
            return extract_qualitative_data_internal(pdf_path)
    except Exception as e:
        raise ToolException(f"PDF parsing failed: {str(e)}")

def extract_financial_metrics_internal(pdf_path: str) -> dict:
    """Internal function to extract metrics from PDF."""
    # TODO: Implement PDF parsing + LLM extraction
    return {"status": "extracted"}
```

### Pattern 1C: Tool with Retry Logic

```python
import time
from functools import wraps
from langchain_core.tools import tool, ToolException

def with_retry(max_retries=3, backoff_base=2):
    """Decorator to add retry logic to tool functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ToolException:
                    # Don't retry validation errors
                    raise
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = backoff_base ** (attempt - 1)
                        print(f"Attempt {attempt} failed. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"All {max_retries} attempts failed")
            
            raise ToolException(f"Failed after {max_retries} retries: {last_exception}")
        
        return wrapper
    return decorator

@tool
@with_retry(max_retries=3, backoff_base=2)
def fetch_market_data(symbol: str) -> dict:
    """
    Fetch TCS market data with automatic retries.
    
    Retry on network errors, not on invalid symbols.
    """
    if symbol != "TCS":
        raise ToolException(f"Unknown symbol: {symbol}")  # Won't retry
    
    # This might fail with network error (will retry)
    import yfinance as yf
    
    try:
        data = yf.Ticker(symbol).info
        return {
            "stock_price": data.get("currentPrice", 0),
            "pe_ratio": data.get("trailingPE", 0),
            "market_cap": data.get("marketCap", 0),
            "timestamp": str(pd.Timestamp.now())
        }
    except ConnectionError:
        raise Exception("Network error - will retry")  # Caught by decorator
```

---

## 2. RAG Setup (Earnings Call Analysis)

### Pattern 2A: Load & Index Earnings Calls

```python
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import AnthropicEmbeddings
from pathlib import Path

def load_and_index_earnings_calls(documents_dir: str) -> PineconeVectorStore:
    """
    Load all earnings call PDFs and index them in Pinecone.
    
    Process:
    1. Load PDFs from directory
    2. Split into chunks
    3. Generate embeddings
    4. Store in Pinecone
    """
    
    # Load all PDFs
    loader = DirectoryLoader(
        documents_dir,
        glob="**/*.pdf",
        loader_cls=PDFPlumberLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Split into chunks (semantic units)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ~200 words per chunk
        chunk_overlap=200,    # Overlap for context
        separators=["\n\n", "\n", ". ", " ", ""]  # Split by these
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # 1536 dimensions
    )
    
    # Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="tcs-earnings-calls",
        namespace="earnings-q3-fy25",
        batch_size=100
    )
    
    print(f"Indexed {len(chunks)} chunks in Pinecone")
    return vectorstore

# Usage
vectorstore = load_and_index_earnings_calls("./data/earnings_calls/")
```

### Pattern 2B: RAG Retriever with Filtering

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def create_hybrid_retriever(vectorstore, documents, top_k=5):
    """
    Create hybrid retriever: semantic (vector) + keyword (BM25).
    
    Benefits:
    - Semantic: "growth" finds "expansion", "momentum"
    - Keyword: "EBITDA" finds exact phrase "EBITDA margin"
    """
    
    # Semantic retriever (vector similarity)
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",  # or "mmr" for diversity
        search_kwargs={"k": top_k}
    )
    
    # Keyword retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k
    
    # Ensemble with weighted combination
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 70% semantic, 30% keyword
    )
    
    return ensemble_retriever

# Advanced: Metadata filtering
def retrieve_with_filters(retriever, query: str, filters: dict) -> list[Document]:
    """
    Retrieve with metadata filters.
    
    Example:
    - filters = {"source": "earnings_q3.pdf", "page": [1, 2, 3]}
    """
    
    results = retriever.get_relevant_documents(query)
    
    # Filter by metadata
    filtered = []
    for doc in results:
        metadata = doc.metadata
        
        # Check all filter conditions
        if all(
            metadata.get(key) in value if isinstance(value, list)
            else metadata.get(key) == value
            for key, value in filters.items()
        ):
            filtered.append(doc)
    
    return filtered
```

### Pattern 2C: RAG-Based Analysis Tool

```python
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import json

@tool
def analyze_earnings_calls(query: str, vectorstore) -> dict:
    """
    Analyze earnings calls using RAG.
    
    Process:
    1. Retrieve relevant chunks from earnings calls
    2. Pass context to LLM
    3. Generate analysis grounded in documents
    """
    
    # Retrieve relevant context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Format context
    context = "\n---\n".join([
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    
    # Create analysis prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a financial analyst analyzing TCS earnings calls.
    
    RETRIEVED CONTEXT:
    {context}
    
    USER QUERY: {query}
    
    ANALYSIS TASK:
    Analyze the earnings calls based on the provided context.
    Return structured JSON with:
    - sentiment: "positive", "neutral", or "negative"
    - key_themes: List of recurring themes
    - management_quotes: 2-3 direct quotes from management
    - risks: Key risks mentioned
    - growth_drivers: Expansion opportunities
    - guidance: Forward-looking statements
    
    IMPORTANT: Only use information from the provided context.
    If information is not in context, say "Not mentioned in calls".
    
    Return ONLY valid JSON, no markdown:
    {{
        "sentiment": "...",
        "key_themes": [...],
        ...
    }}
    """)
    
    # Create chain
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.2  # Low randomness for consistent analysis
    )
    
    chain = prompt | llm
    
    # Invoke
    response = chain.invoke({
        "context": context,
        "query": query
    })
    
    # Parse JSON
    result = json.loads(response.content)
    result["source_count"] = len(retrieved_docs)
    result["retrieved_sources"] = [
        doc.metadata.get("source", "unknown")
        for doc in retrieved_docs
    ]
    
    return result
```

---

## 3. Agent with LangGraph

### Pattern 3A: Complete ReAct Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from typing import Annotated, TypedDict
import uuid

# Define agent state
class ForecastState(TypedDict):
    """State maintained throughout agent execution."""
    
    # Primary inputs/outputs
    query: str
    forecast_result: dict
    
    # Intermediate data
    financial_metrics: dict
    qualitative_analysis: dict
    market_data: dict
    
    # Metadata
    messages: list[BaseMessage]
    tool_calls: list[dict]
    errors: list[str]

def create_tcs_forecast_agent():
    """Create ReAct agent for TCS forecasting."""
    
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )
    
    # Define tools (from patterns above)
    tools = [
        extract_financial_metrics,
        analyze_earnings_calls,
        fetch_market_data
    ]
    
    # Create memory for persistence
    memory = MemorySaver()
    
    # Create ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        state_modifier="""You are a TCS financial forecasting agent.

Your task: Generate a business outlook forecast for TCS's next quarter.

Process:
1. Extract financial metrics from latest quarterly reports
2. Analyze management outlook from earnings calls
3. Check current market conditions
4. Synthesize into a comprehensive forecast

Requirements:
- All claims must reference source documents
- Include confidence scores (0.0-1.0) for each prediction
- Highlight key risks and uncertainties
- Return structured JSON forecast

Format your final answer as JSON matching the required schema."""
    )
    
    return agent

# Usage
agent = create_tcs_forecast_agent()

# Execute agent
config = {"configurable": {"thread_id": f"forecast_{uuid.uuid4().hex[:8]}"}}

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Generate TCS forecast for Q4 FY25"
    }]
}, config=config)

print(result["messages"][-1].content)
```

### Pattern 3B: Custom Node-Based Agent (Advanced)

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
import json

def create_custom_agent():
    """Create agent with explicit nodes for more control."""
    
    # Create graph
    graph = StateGraph(ForecastState)
    
    # Node 1: Planning
    def planning_node(state: ForecastState) -> ForecastState:
        """Agent decides which tools to use."""
        state["messages"].append(
            AIMessage("Starting forecast generation. Will extract metrics first...")
        )
        return state
    
    # Node 2: Metric Extraction
    def extraction_node(state: ForecastState) -> ForecastState:
        """Execute metric extraction tool."""
        try:
            metrics = extract_financial_metrics(
                "Load from recent report..."
            )
            state["financial_metrics"] = metrics
        except Exception as e:
            state["errors"].append(f"Extraction failed: {e}")
        
        return state
    
    # Node 3: Qualitative Analysis
    def analysis_node(state: ForecastState) -> ForecastState:
        """Execute earnings call analysis tool."""
        try:
            analysis = analyze_earnings_calls(
                "What is management's outlook?",
                vectorstore
            )
            state["qualitative_analysis"] = analysis
        except Exception as e:
            state["errors"].append(f"Analysis failed: {e}")
        
        return state
    
    # Node 4: Synthesis
    def synthesis_node(state: ForecastState) -> ForecastState:
        """Combine all data into forecast."""
        
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        
        synthesis_prompt = f"""
        Based on:
        - Financial metrics: {state['financial_metrics']}
        - Qualitative analysis: {state['qualitative_analysis']}
        - Market data: {state['market_data']}
        
        Generate a TCS Q4 FY25 forecast.
        
        Return JSON:
        {{
            "forecast_summary": "...",
            "expected_growth": "...",
            "confidence": 0.0-1.0,
            "key_assumptions": [...],
            "risks": [...]
        }}
        """
        
        response = llm.invoke(synthesis_prompt)
        state["forecast_result"] = json.loads(response.content)
        
        return state
    
    # Add nodes to graph
    graph.add_node("planning", planning_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("synthesis", synthesis_node)
    
    # Define edges
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "extraction")
    graph.add_edge("extraction", "analysis")
    graph.add_edge("analysis", "synthesis")
    graph.add_edge("synthesis", END)
    
    # Compile with memory
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph

# Usage
agent = create_custom_agent()

initial_state = ForecastState(
    query="Generate TCS forecast",
    forecast_result={},
    financial_metrics={},
    qualitative_analysis={},
    market_data={},
    messages=[HumanMessage("Start forecast")],
    tool_calls=[],
    errors=[]
)

result = agent.invoke(initial_state)
print("Forecast:", result["forecast_result"])
```

---

## 4. Error Handling Patterns

### Pattern 4A: Tool-Level Error Handling

```python
from langchain_core.tools import tool, ToolException
import logging

logger = logging.getLogger(__name__)

@tool
def robust_extractor(pdf_path: str) -> dict:
    """Extract with comprehensive error handling."""
    
    # Pre-execution validation
    if not pdf_path:
        raise ToolException("PDF path cannot be empty")
    
    try:
        # Main logic
        result = _extract_from_pdf(pdf_path)
        
        # Validate result
        if not result or len(result) == 0:
            raise ToolException("No metrics extracted from PDF")
        
        return result
        
    except FileNotFoundError:
        logger.error(f"PDF not found: {pdf_path}")
        raise ToolException(f"File not found: {pdf_path}")
    
    except ValueError as e:
        logger.error(f"Invalid PDF format: {e}")
        raise ToolException(f"Invalid PDF format: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected extraction error: {e}", exc_info=True)
        raise ToolException(f"Extraction failed: {str(e)}")
```

### Pattern 4B: Agent-Level Error Recovery

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import ToolException

def create_resilient_agent():
    """Agent that continues despite tool failures."""
    
    graph = StateGraph(ForecastState)
    
    def extraction_with_fallback(state: ForecastState):
        """Try extraction, continue with empty data if it fails."""
        try:
            state["financial_metrics"] = extract_financial_metrics(...)
        except ToolException as e:
            logger.warning(f"Extraction failed: {e}, continuing...")
            state["financial_metrics"] = {}
            state["errors"].append(str(e))
        
        return state
    
    def synthesis_with_partial_data(state: ForecastState):
        """Synthesize forecast even with incomplete data."""
        
        metrics = state.get("financial_metrics", {})
        analysis = state.get("qualitative_analysis", {})
        
        # Alert if data is missing
        if not metrics:
            logger.warning("Proceeding without financial metrics")
        if not analysis:
            logger.warning("Proceeding without qualitative analysis")
        
        # Generate partial forecast
        synthesis_prompt = f"""
        Generate forecast with available data:
        {{"metrics_available": bool(metrics), "analysis_available": bool(analysis)}}
        
        Note: Some data is missing. Do your best with available information.
        """
        
        # Continue synthesis...
        return state
    
    graph.add_node("extraction", extraction_with_fallback)
    graph.add_node("synthesis", synthesis_with_partial_data)
    
    # Compile and return
    return graph.compile()
```

---

## 5. FastAPI Integration

### Pattern 5A: Simple Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ForecastRequest(BaseModel):
    quarter: str  # e.g., "Q4 FY25"
    include_market_data: bool = True

class ForecastResponse(BaseModel):
    forecast_summary: str
    expected_growth: float
    confidence: float
    risks: list[str]
    source_documents: list[str]

agent = create_tcs_forecast_agent()

@app.post("/forecast")
async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate TCS business forecast."""
    
    try:
        # Invoke agent (blocking, so run in thread pool)
        result = await asyncio.to_thread(
            agent.invoke,
            {
                "messages": [{
                    "role": "user",
                    "content": f"Generate TCS forecast for {request.quarter}"
                }]
            }
        )
        
        # Extract forecast from result
        forecast_json = parse_agent_response(result)
        
        # Log to database
        await log_forecast_async(request.dict(), forecast_json)
        
        return ForecastResponse(**forecast_json)
        
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        await log_error_async(request.dict(), str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

### Pattern 5B: Async Integration

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Create thread pool for blocking LLM calls
executor = ThreadPoolExecutor(max_workers=5)

async def generate_forecast_async(request: ForecastRequest):
    """Non-blocking forecast generation."""
    
    # Run agent in thread pool
    loop = asyncio.get_event_loop()
    
    result = await loop.run_in_executor(
        executor,
        lambda: agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Generate {request.quarter} forecast"
            }]
        })
    )
    
    # Process result
    forecast = parse_result(result)
    
    # Log asynchronously in background
    asyncio.create_task(
        log_forecast_async(request.dict(), forecast)
    )
    
    return forecast
```

---

## 6. Testing & Validation

### Pattern 6A: Unit Test for Tool

```python
import pytest
from unittest.mock import patch, MagicMock

def test_extract_financial_metrics():
    """Test metric extraction tool."""
    
    # Test data
    sample_report = """
    TCS Q3 FY25 Financial Results:
    - Total Revenue: ₹60,000 crores
    - Net Profit: ₹12,000 crores
    - Operating Margin: 20.5%
    """
    
    # Mock LLM
    with patch('langchain_anthropic.ChatAnthropic.invoke') as mock_llm:
        mock_llm.return_value.content = json.dumps({
            "revenue": {"value": "₹60,000 crores", "confidence": 0.95},
            "net_profit": {"value": "₹12,000 crores", "confidence": 0.95}
        })
        
        # Test
        result = extract_financial_metrics(sample_report)
        
        # Assert
        assert "revenue" in result["metrics"]
        assert result["metrics"]["revenue"]["confidence"] >= 0.6
        assert len(result["metrics"]) >= 2
```

### Pattern 6B: Integration Test

```python
def test_full_forecast_generation():
    """Test complete forecast generation."""
    
    agent = create_tcs_forecast_agent()
    
    # Test with mock documents
    with patch.object(agent, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [MagicMock(content=json.dumps({
                "forecast_summary": "Expected 8-10% growth",
                "confidence": 0.85,
                "risks": ["Geopolitical tension", "Currency volatility"]
            }))]
        }
        
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Generate forecast"
            }]
        })
        
        # Validate response structure
        forecast = json.loads(result["messages"][0].content)
        assert "forecast_summary" in forecast
        assert "confidence" in forecast
        assert forecast["confidence"] >= 0
        assert forecast["confidence"] <= 1
```

---

## Quick Reference: Copy-Paste Templates

### Template 1: Minimal Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_data(query: str) -> str:
    """Fetch data."""
    return "Sample data"

agent = create_react_agent(
    ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    [get_data]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Your query"}]
})
```

### Template 2: RAG Retriever

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = PineconeVectorStore.from_existing_index(
    index_name="your-index",
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("your query")
```

### Template 3: Error-Safe Tool

```python
@tool
def safe_tool(input_data: str) -> dict:
    """Safe tool with error handling."""
    try:
        # Your logic
        return {"result": "success"}
    except Exception as e:
        raise ToolException(f"Failed: {e}")
```

---

**Use these patterns as building blocks for your TCS Forecast Agent implementation!**
