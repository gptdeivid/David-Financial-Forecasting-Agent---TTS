# 🗺️ LangChain Visual Reference Guide

**Visual quick-reference for complex concepts**
**Format**: ASCII diagrams, decision trees, architecture flows
**Purpose**: Understand at a glance

---

## VISUAL 1: Complete LangChain Stack

```
┌────────────────────────────────────────────────────────────┐
│                  USER LAYER                               │
│              FastAPI Endpoint (POST /forecast)            │
└────────────────────┬─────────────────────────────────────┘
                     │ {"query": "Generate TCS forecast"}
                     ▼
┌────────────────────────────────────────────────────────────┐
│           LANGCHAIN / LANGGRAPH LAYER                     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │  ReAct Agent (LangGraph)                         │    │
│  │  ├─ State Machine (Reason → Act → Observe)      │    │
│  │  ├─ Tool Registry (3 tools)                      │    │
│  │  ├─ Checkpointer (MySQL)                        │    │
│  │  └─ Message History                             │    │
│  └──────────────────────────────────────────────────┘    │
│                     │                                     │
│        ┌────────────┼────────────┐                       │
│        │            │            │                       │
│        ▼            ▼            ▼                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │  TOOL 1  │ │  TOOL 2  │ │  TOOL 3  │              │
│  │ Extractor│ │ Analyzer │ │ Market   │              │
│  └──────────┘ └──────────┘ └──────────┘              │
│                                                            │
│  LLM Integration (Temperature-controlled)                │
│  ├─ Extraction: temp=0.0 (deterministic)                │
│  ├─ Analysis: temp=0.2 (consistent)                     │
│  └─ Synthesis: temp=0.4 (balanced)                      │
│                                                            │
│  Memory & State                                          │
│  ├─ Conversation history                                │
│  ├─ Intermediate results                                │
│  ├─ Tool execution trace                                │
│  └─ Confidence scores                                   │
└────────────────────────────────────────────────────────────┘
                     │
        ┌────────────┼──────────────┐
        ▼            ▼              ▼
┌───────────────┐ ┌──────────┐ ┌──────────┐
│  LLM APIs     │ │ Vector   │ │ External │
│ (Claude,      │ │ Stores   │ │ Services │
│  GPT-4)       │ │(Pinecone)│ │ (Finance)│
└───────────────┘ └──────────┘ └──────────┘
        │            │              │
        ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│           EXTERNAL SERVICES                             │
│  ├─ Anthropic/OpenAI APIs (LLM inference)              │
│  ├─ Pinecone (earnings calls vector store)             │
│  ├─ MySQL (logging & persistence)                      │
│  └─ Yahoo Finance (market data)                        │
└─────────────────────────────────────────────────────────┘
```

---

## VISUAL 2: ReAct Agent Execution Flow

```
        ┌─────────────────────────────────┐
        │   User Query                    │
        │  "Generate TCS forecast"        │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [REASON]                       │
        │  Agent analyzes:                │
        │  • Need financial metrics       │
        │  • Need management outlook      │
        │  • Need market conditions       │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [ACT] Call Tool 1              │
        │  extract_financial_metrics()    │
        │  → Input: Q3 FY25 report PDF    │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [OBSERVE] Receive Tool Output  │
        │  {                              │
        │    "revenue": 60000,            │
        │    "profit": 12000,             │
        │    "margin": 0.20               │
        │  }                              │
        │  Update: agent_scratchpad       │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [REASON] (2nd iteration)       │
        │  • Metrics acquired ✓           │
        │  • Still need: sentiment        │
        │  → Use Tool 2                   │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [ACT] Call Tool 2              │
        │  analyze_earnings_calls()       │
        │  → Query: "management outlook"  │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [OBSERVE] Receive Tool Output  │
        │  {                              │
        │    "sentiment": "positive",     │
        │    "themes": [                  │
        │      "digital transformation"   │
        │    ]                            │
        │  }                              │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [REASON] (3rd iteration)       │
        │  • Metrics acquired ✓           │
        │  • Sentiment acquired ✓         │
        │  • Ready to synthesize!         │
        │  → Final answer generation      │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │  [FINAL ANSWER]                 │
        │  {                              │
        │    "forecast_summary": "...",   │
        │    "expected_growth": "8-10%",  │
        │    "confidence": 0.85,          │
        │    "risks": [...]               │
        │  }                              │
        └─────────────────────────────────┘
```

---

## VISUAL 3: Tool Execution Pipeline

```
INPUT: Financial Report PDF
  │
  ▼
┌─────────────────────────────┐
│  TOOL 1: Metric Extraction  │
│  Temperature: 0.0           │
│  Type: Deterministic        │
│                             │
│  Process:                   │
│  1. PDF → text extraction   │
│  2. LLM extraction prompt   │
│  3. JSON parsing            │
│  4. Confidence filtering    │
│  5. Source validation       │
└──────────────┬──────────────┘
               │
    ┌──────────▼──────────┐
    │  Result: {          │
    │    revenue: {...},  │
    │    profit: {...},   │
    │    margins: {...}   │
    │  }                  │
    └──────────┬──────────┘
               │
               ▼
INPUT: Earnings Call Transcript
  │
  ▼
┌─────────────────────────────┐
│  TOOL 2: Qualitative        │
│  Analysis (RAG)             │
│  Temperature: 0.2           │
│  Type: Semantic Search      │
│                             │
│  Process:                   │
│  1. Query embedding         │
│  2. Vector search (Top-3)   │
│  3. Context retrieval       │
│  4. LLM analysis            │
│  5. Sentiment scoring       │
└──────────┬──────────────────┘
           │
    ┌──────▼────────┐
    │ Result: {     │
    │  sentiment: "positive",
    │  themes: [...]
    │ }             │
    └──────┬────────┘
           │
           ▼
INPUT: Current Stock Data
  │
  ▼
┌─────────────────────────────┐
│  TOOL 3: Market Data        │
│  Temperature: N/A           │
│  Type: API Call             │
│                             │
│  Process:                   │
│  1. Call Yahoo Finance API  │
│  2. Parse stock price       │
│  3. Compute ratios          │
│  4. Validate data           │
└──────────┬──────────────────┘
           │
    ┌──────▼────────┐
    │ Result: {     │
    │  price: 3800, │
    │  pe: 28.5     │
    │ }             │
    └──────┬────────┘
           │
           ▼
      ┌─────────────┐
      │ All Inputs  │
      │ Combined    │
      └────┬────────┘
           │
           ▼
┌─────────────────────────────┐
│  SYNTHESIS (TOOL 4)         │
│  Temperature: 0.4           │
│  Type: LLM Generation       │
│                             │
│  Process:                   │
│  1. Merge all data          │
│  2. Validate consistency    │
│  3. Generate forecast       │
│  4. Add confidence scores   │
│  5. Format JSON response    │
└──────────┬──────────────────┘
           │
           ▼
OUTPUT: Forecast JSON
{
  "forecast_summary": "TCS expected to grow 8-10%...",
  "key_financial_trends": [...],
  "management_outlook": "Positive on digital growth",
  "risks_and_opportunities": [...],
  "market_data": {...},
  "source_documents": [...],
  "errors": []
}
```

---

## VISUAL 4: RAG System Architecture

```
DOCUMENT INDEXING (One-time setup)
────────────────────────────────────

    TCS Financial Report.pdf
    ├─ Load
    ├─ Split into chunks (1000 chars)
    ├─ Generate embeddings
    │  └─ "Revenue exceeded expectations"
    │     → [0.23, -0.45, 0.67, ..., -0.89] (1536 dims)
    │
    └─ Store in Vector DB (Pinecone)
       ├─ Chunk 1: [vector] + metadata
       ├─ Chunk 2: [vector] + metadata
       └─ Chunk N: [vector] + metadata


RETRIEVAL (Per query)
────────────────────────────────────

    User Query: "What is management's view on growth?"
    │
    ├─ Embedding Generation
    │  └─ Query → [0.21, -0.43, 0.65, ..., -0.87] (same model)
    │
    ├─ Vector Search (Similarity)
    │  Query Vector ──┐
    │                 ├─→ Compare with all stored vectors
    │  Document 1 ───┤
    │  Document 2 ───┤
    │  Document 3 ───┤
    │  ...            │
    │  Document N ───┘
    │
    ├─ Top-3 Most Similar
    │  1. [Chunk 45] Similarity: 0.92
    │  2. [Chunk 23] Similarity: 0.88
    │  3. [Chunk 67] Similarity: 0.81
    │
    └─ Return Context
       "Management stated growth drivers include..."


GENERATION
────────────────────────────────────

    LLM Prompt:
    ┌─────────────────────────────────────────┐
    │ Context (from retrieval):               │
    │ "Management stated growth drivers       │
    │  include digital transformation..."     │
    │                                         │
    │ Question: What is management view?     │
    │                                         │
    │ Answer: (generated grounded in context)│
    │ "According to management, growth is     │
    │  driven by digital transformation..."  │
    └─────────────────────────────────────────┘
```

---

## VISUAL 5: Error Handling Decision Tree

```
                    Tool Called
                         │
                         ▼
                  Does it Complete?
                    /            \
                  YES            NO
                  │              │
                  ▼              ▼
            Return Result    What Error?
                             /    |    \
                            /     |     \
                    Validation  Network  Logic
                      Error     Error    Error
                       │         │       │
                       │         │       │
                      Don't    Retry    Log &
                      Retry   (x3)      Continue
                       │       │         │
                       ▼       ▼         ▼
                   Raise   Wait 2s,  Partial
                Exception  4s, 8s    Forecast
                       │      │        │
                       │      │        │
                       └──────┴────────┘
                            │
                            ▼
                    Agent Continues
                    with Other Tools
```

---

## VISUAL 6: Temperature Impact

```
TEMPERATURE SCALE
(0.0 = deterministic, 1.0 = random)

Temperature = 0.0 (EXTRACTION)
┌──────────────────────────────┐
│ "Revenue is 60,000 crores"   │
│ ✓ Always same output         │
│ ✓ Deterministic              │
│ ✗ No creativity              │
└──────────────────────────────┘
          ▲
          │ (more deterministic)
          │

Temperature = 0.2 (ANALYSIS)
┌──────────────────────────────┐
│ Output A: "Sentiment: positive"
│ Output B: "Sentiment: positive"
│ Output C: "Sentiment: positive"
│ ✓ Consistent interpretation  │
│ ✗ Some variation possible    │
└──────────────────────────────┘
          ▲
          │ (more diverse)
          │

Temperature = 0.4 (SYNTHESIS)
┌──────────────────────────────┐
│ Output A: "Growth 8-10%..."  │
│ Output B: "Growth 9-11%..."  │
│ Output C: "Growth 7-9%..."   │
│ ✓ Balanced creativity        │
│ ✓ Reasonable variation       │
└──────────────────────────────┘
          ▲
          │ (more random)
          │

Temperature = 1.0+ (TOO RANDOM)
┌──────────────────────────────┐
│ Output A: "Growth 15%..."    │
│ Output B: "Decline 5%..."    │
│ Output C: "Revenue doubled"  │
│ ✗ Inconsistent/hallucinated │
│ ✗ Unreliable                │
└──────────────────────────────┘
```

---

## VISUAL 7: Agent State Management

```
INITIAL STATE
┌─────────────────────────────────┐
│ Query: "Generate forecast"      │
│ Financial_metrics: {}           │
│ Qualitative_analysis: {}        │
│ Market_data: {}                 │
│ Errors: []                      │
│ Tool_calls: []                  │
└─────────────────────────────────┘
         │
         ▼ Tool 1 Executes
┌─────────────────────────────────┐
│ Query: "Generate forecast"      │
│ Financial_metrics: {            │
│   revenue: 60000,               │
│   profit: 12000                 │
│ }                               │
│ Qualitative_analysis: {}        │
│ Market_data: {}                 │
│ Errors: []                      │
│ Tool_calls: [exec_tool_1]       │
└─────────────────────────────────┘
         │
         ▼ Tool 2 Executes
┌─────────────────────────────────┐
│ Query: "Generate forecast"      │
│ Financial_metrics: {...}        │
│ Qualitative_analysis: {         │
│   sentiment: "positive"         │
│ }                               │
│ Market_data: {}                 │
│ Errors: []                      │
│ Tool_calls: [exec_1, exec_2]    │
└─────────────────────────────────┘
         │
         ▼ Tool 3 Executes
┌─────────────────────────────────┐
│ Query: "Generate forecast"      │
│ Financial_metrics: {...}        │
│ Qualitative_analysis: {...}     │
│ Market_data: {                  │
│   price: 3800, pe_ratio: 28.5   │
│ }                               │
│ Errors: []                      │
│ Tool_calls: [exec_1, 2, 3]      │
└─────────────────────────────────┘
         │
         ▼ Synthesis
┌─────────────────────────────────┐
│ FINAL STATE                     │
│ Forecast_result: {              │
│   summary: "8-10% growth",      │
│   confidence: 0.85              │
│ }                               │
│ All previous state + result     │
└─────────────────────────────────┘
         │
         ▼ PERSISTED TO MYSQL
    For audit & recovery
```

---

## VISUAL 8: LangChain vs LangGraph Comparison

```
DECISION MATRIX
═══════════════════════════════════════════════════════════

                    LangChain       LangGraph
                    (LCEL)          (Modern)
                    ────────        ─────────

Single-step         ✓✓              ✓
chains              Easy            Easy

Multi-tool          ✓ OK            ✓✓
agents              Works           Better

State               ✗               ✓✓
persistence         Manual          Automatic

Error               ✓               ✓✓
recovery            Basic           Advanced

Flow                ✓               ✓✓
control             Limited         Full

Production          ✓               ✓✓
readiness           OK              Recommended

Debugging           ✓               ✓✓
                    Moderate        Excellent

Learning            ✓               ✓✓
curve                Easy            Medium


USE CASES:
─────────────────────────────────────────────────────────

LangChain LCEL:
├─ Simple prompt → LLM → parse
├─ Single tool
└─ Fast prototyping

LangGraph:
├─ Multi-step reasoning
├─ Persistent state
├─ Error recovery
├─ Production systems
└─ TCS Forecast Agent ← USE THIS
```

---

## VISUAL 9: Common Failure Modes & Recovery

```
FAILURE CASCADE & RECOVERY

1. TOOL EXECUTION FAILS
   ├─ PDF parsing error
   ├─ LLM API timeout
   └─ Invalid data format
        │
        ▼
   Tool Exception → Log error → Continue?
        │
        ├─ YES: Agent uses other tools (graceful degradation)
        │       Result: Partial forecast (better than none)
        │
        └─ NO: Return error
                Result: No forecast (worst case)

2. VALIDATION FAILS
   ├─ Low confidence (<0.6)
   ├─ Invalid metric values
   └─ Inconsistent data
        │
        ▼
   Filter bad data → Log warning → Continue
        │
        └─ Agent aware of data quality
            Result: Forecast with caveats

3. ALL TOOLS FAIL
   ├─ API downtime
   ├─ Vector store unavailable
   └─ LLM unreachable
        │
        ▼
   Checkpoint restore (replay from step 2)
        │
        └─ If still fails: Return cached forecast
            Result: Stale but available
```

---

## VISUAL 10: Performance Targets

```
LATENCY BREAKDOWN (Per Forecast)
────────────────────────────────────

Tool 1 (Extraction):  5 seconds
├─ PDF load: 1s
├─ LLM call: 3s
└─ Parse: 1s

Tool 2 (Analysis):    8 seconds
├─ Vector search: 1s
├─ RAG retrieval: 2s
└─ LLM analysis: 5s

Tool 3 (Market):      2 seconds
├─ API call: 1s
└─ Parse: 1s

Tool 4 (Synthesis):   5 seconds
├─ Merge data: 1s
├─ LLM generation: 3s
└─ Format: 1s

Database Logging:     1 second (async)
────────────────────────────────────
TOTAL:               ~20 seconds
TARGET:              <30 seconds ✓


THROUGHPUT
────────────────────────────────────

Sequential:
  1 agent × 20 seconds = 5 forecasts/minute

Parallel (ideal):
  3 agents × 20 seconds = 9 forecasts/minute

With async I/O:
  10+ forecasts/minute ✓


ERROR RATES
────────────────────────────────────

Tool success rate:        95% ✓
Agent completion:         98% ✓
API availability:         99.5% ✓
Database logging:         99.9% ✓
Overall service:          98% ✓
```

---

**Use these visuals as reference while coding and during team discussions!**
