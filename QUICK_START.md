# TCS Financial Forecasting Agent - Quick Start Guide

## 🚀 5-Minute Overview

This project builds an **AI-powered Financial Forecasting Agent** for TCS that:

- Automatically analyzes financial reports and earnings calls
- Extracts quantitative metrics (revenue, margins, growth)
- Performs qualitative analysis (management sentiment, themes, risks)
- Generates structured business outlook forecasts
- Maintains complete audit trail in MySQL

**Tech Stack**: FastAPI + LangChain + Gemini/Claude + Chroma/Pinecone + MySQL

---

## 📋 Quick Start (3 Steps)

### Step 1: Environment Setup

```bash
# Clone and setup
git clone <your-repo>
cd tcs-forecast-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env with your API keys
```

### Step 2: Initialize Infrastructure

```bash
# Setup MySQL
mysql -u root -p < database/init.sql

# Initialize vector store and ingest documents
python scripts/init_vector_store.py
python scripts/download_documents.py
python scripts/ingest_documents.py
```

### Step 3: Run Application

```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"company": "TCS"}'

# View API docs at http://localhost:8000/docs
```

---

## 🏗️ Architecture Overview

```
CLIENT REQUEST
    ↓
FASTAPI (Async)
    ↓
LANGCHAIN REACT AGENT
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
FINANCIAL         QUALITATIVE        MARKET DATA
EXTRACTOR         ANALYSIS           (Optional)
(LLM + PDF)       (RAG + LLM)        (API)
│                 │                  │
└─────────────────┴──────────────────┘
    ↓
SYNTHESIS & REASONING
    ↓
JSON RESPONSE + MySQL LOG
```

**Key Components**:

1. **Document Pipeline**: Scrape → Parse → Chunk → Embed
2. **Vector Store**: Semantic search across earnings calls
3. **LLM Tools**: Financial extraction + Qualitative analysis
4. **Agent**: ReAct pattern orchestrates tools
5. **API**: FastAPI with async logging

---

## 🛠️ Development Phases (7 Weeks)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Foundation | FastAPI skeleton, MySQL setup |
| 2 | Document Pipeline | PDF scraper, parser, chunking |
| 3 | RAG & Vector Store | Chroma setup, embedding pipeline |
| 4 | Tool Implementation | FinancialExtractor, QualitativeAnalysis |
| 5 | Agent Orchestration | LangChain ReAct agent |
| 6 | Testing & QA | Full test suite, accuracy evaluation |
| 7 | Deployment | Production deployment, documentation |

---

## 🔑 Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Gemini 1.5 Pro / Claude 3.5 | 2M context, PDF vision, financial reasoning |
| **Embeddings** | text-embedding-3-small | Fast, cost-effective |
| **Vector DB** | Chroma (dev) / Pinecone (prod) | Easy local dev, scalable prod |
| **Framework** | LangChain | Best tool ecosystem |
| **Web Server** | FastAPI | Async-first, auto-docs |
| **Database** | MySQL 8.0 | Per requirements, JSON support |

---

## 📊 AI Reasoning Approach

### ReAct Pattern (Reason → Act → Observe)

```
USER: "Generate Q4 FY25 forecast for TCS"
    ↓
AGENT THINKS: "I need Q3 metrics + management sentiment"
    ↓
AGENT ACTS:
  - Tool 1: Extract metrics from Q3 report
  - Tool 2: Analyze earnings call for themes
    ↓
AGENT OBSERVES:
  - Revenue: +5.4% YoY, Margin: 21%
  - Sentiment: Cautious on client spending
    ↓
AGENT SYNTHESIZES:
  "Q4 growth moderates to 3-5% due to client
   caution, offset by AI services momentum"
```

### Master Prompts

**Financial Extraction** (Temperature: 0.0)

- Extract ONLY explicitly stated metrics
- Require source quotes for verification
- Assign confidence scores (0-1)
- Validate mathematical consistency

**Qualitative Analysis** (Temperature: 0.2)

- Identify themes from earnings calls
- Extract sentiment and direct quotes
- Assess forecast relevance
- Flag recurring mentions

**Synthesis** (Temperature: 0.4)

- Cross-validate quant vs qual signals
- Detect and resolve conflicts
- Generate forecast with rationale
- Cite all sources

---

## 🔒 Guardrails & Quality

### 1. Source Verification

Every extracted metric verified against source text:

```python
if metric["source_quote"] not in document_text:
    mark_as_hallucination()
    exclude_from_forecast()
```

### 2. Confidence Scoring

```
1.0 = Explicitly in bold/table
0.8 = Clearly in body text
0.6 = Requires interpretation
<0.6 = EXCLUDED
```

### 3. Consistency Checks

- Revenue > 0
- Margin = Profit / Revenue (±2% tolerance)
- Growth rate reasonable (-50% to +50%)

### 4. Error Recovery

- Retry with exponential backoff (3 attempts)
- Fallback to partial forecast if tools fail
- All errors logged to MySQL

---

## 📁 Project Structure

```
tcs-forecast-agent/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── config.py                  # Settings
│   ├── models/                    # Pydantic schemas
│   ├── services/
│   │   ├── agent.py               # LangChain agent
│   │   ├── document_processor.py  # PDF parsing
│   │   ├── vector_store.py        # RAG
│   │   ├── logger.py              # MySQL logger
│   │   └── tools/                 # Specialized tools
│   └── routes/                    # API endpoints
├── scripts/
│   ├── download_documents.py      # Fetch TCS docs
│   ├── ingest_documents.py        # Populate vector store
│   └── test_agent.py              # Manual testing
├── database/
│   └── init.sql                   # MySQL schema
├── tests/                         # Test suite
├── data/                          # Downloaded PDFs
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🎯 Success Metrics

### Technical

- API Latency: < 30s (p95)
- Success Rate: > 95%
- Hallucination Rate: < 5%
- Source Fidelity: > 90%

### Business

- Forecast Accuracy (Revenue MAPE): < 10%
- Direction Accuracy: > 70%
- Time Savings: 80% vs manual (2h → 0.5h)

---

## 🚨 Common Issues & Solutions

**Issue**: PDF parsing fails

- **Solution**: Use Gemini Vision API for complex layouts

**Issue**: LLM hallucinates metrics

- **Solution**: Enforce structured output, require source quotes, use temp=0.0

**Issue**: Vector search returns irrelevant segments

- **Solution**: Adjust chunk size, add metadata filters, use hybrid search

**Issue**: API timeout on first request

- **Solution**: Pre-warm services on startup, cache documents

**Issue**: MySQL logging slows responses

- **Solution**: Use background tasks (already implemented)

---

## 📦 Deployment Options

### Local Development

```bash
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
docker-compose up -d
```

### Cloud (AWS Example)

- **Compute**: ECS Fargate
- **Database**: RDS MySQL
- **Vector Store**: Pinecone
- **Storage**: S3
- **Monitoring**: CloudWatch

---

## 📚 Key Files to Review

1. **IMPLEMENTATION_PLAN.md**: Complete 10-section implementation guide
2. **app/main.py**: FastAPI application entry point
3. **app/services/agent.py**: LangChain ReAct agent
4. **app/services/tools/**: Specialized analysis tools
5. **database/init.sql**: MySQL schema
6. **requirements.txt**: All dependencies

---

## 🔄 Development Workflow

```bash
# Terminal 1: Start server with auto-reload
uvicorn app.main:app --reload

# Terminal 2: Run tests
pytest tests/ -v

# Terminal 3: Monitor logs
tail -f app.log
```

---

## ✅ Pre-Production Checklist

- [ ] All dependencies pinned in requirements.txt
- [ ] .env.example created (no secrets committed)
- [ ] Database schema executed
- [ ] Vector store initialized and documents ingested
- [ ] All tests passing
- [ ] Health endpoint returning 200
- [ ] Forecast endpoint working
- [ ] MySQL logging confirmed
- [ ] Error scenarios handled
- [ ] Response schema validated
- [ ] Performance acceptable (< 30s)
- [ ] Documentation complete

---

## 📞 Getting Help

1. Check `app.log` for errors
2. Query MySQL logs: `SELECT * FROM api_request_logs WHERE status='error' LIMIT 10;`
3. Verify API credentials in `.env`
4. Check vector store status
5. Review LangChain debug output (set `verbose=True`)

---

## 📖 Additional Resources

- **Full Implementation Plan**: See `IMPLEMENTATION_PLAN.md`
- **LangChain Docs**: <https://python.langchain.com/docs/>
- **FastAPI Docs**: <https://fastapi.tiangolo.com/>
- **Gemini API**: <https://ai.google.dev/docs>

---

**Ready to build? Start with Phase 1 in IMPLEMENTATION_PLAN.md!** 🚀
