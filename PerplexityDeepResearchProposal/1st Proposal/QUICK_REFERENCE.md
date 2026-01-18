# TCS Forecast Agent: Implementation Summary & Quick Reference

## 📋 What This Guide Delivers

You now have:

1. **Comprehensive 10-section guide** covering architecture, tools, setup, and deployment
2. **AI stack specifications** with reasoning approaches and guardrails
3. **Complete starter code** with FastAPI app, config, and logging
4. **requirements.txt** with all dependencies versioned
5. **Database schema** and async logging implementation
6. **Error handling patterns** and recovery strategies
7. **Evaluation framework** with quality metrics
8. **Real-world examples** and troubleshooting scenarios

---

## 🚀 Quick Start (5 Steps)

### Step 1: Environment Setup (5 minutes)
```bash
# Clone and activate
git clone <your-repo>
cd tcs-forecast-agent
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Secrets (3 minutes)
```bash
# Copy template
cp .env.example .env

# Edit .env with your API keys:
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=...
# MYSQL_PASSWORD=...
# etc.
```

### Step 3: Initialize Infrastructure (10 minutes)
```bash
# Setup MySQL
mysql -u root -p < database/init.sql

# Initialize Pinecone index
python scripts/init_vector_store.py

# Ingest financial documents
python scripts/ingest_documents.py
```

### Step 4: Start Service (1 minute)
```bash
# Terminal window 1: Start FastAPI
uvicorn app.main:app --reload --port 8000

# Output: Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Test Endpoint (2 minutes)
```bash
# Terminal window 2: Test forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"company": "TCS", "include_market_data": true}'

# Check API docs at http://localhost:8000/docs
```

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLIENT REQUEST                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI ENDPOINT                            │
│  POST /forecast {"company": "TCS"}                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              LANGCHAIN REACT AGENT (Orchestrator)               │
│  - Decides which tools to call                                  │
│  - Chains tool outputs                                          │
│  - Synthesizes forecast                                         │
└──────────┬──────────────────────────────────────────────────────┘
           │
      ┌────┴─────────────────────────────┐
      │                                  │
      ▼                                  ▼
┌────────────────────┐          ┌────────────────────┐
│ FINANCIAL DATA     │          │ QUALITATIVE        │
│ EXTRACTOR TOOL     │          │ ANALYSIS TOOL      │
│                    │          │                    │
│ Input: PDF Report  │          │ Input: Query       │
│ ↓                  │          │ ↓                  │
│ LLM Extraction     │          │ Embed Query        │
│ ↓                  │          │ ↓                  │
│ Metrics Output     │          │ Vector Search      │
│                    │          │ ↓                  │
│ - Revenue: 62.6Cr  │          │ Retrieve Segments  │
│ - Margin: 21%      │          │ ↓                  │
│ - Growth: +5.4%    │          │ LLM Analysis       │
│                    │          │ ↓                  │
│                    │          │ Insights Output    │
│                    │          │                    │
│                    │          │ - Sentiment: +     │
│                    │          │ - Themes: Growth   │
│                    │          │ - Risks: Caution   │
└────┬───────────────┘          └────┬───────────────┘
     │                              │
     └──────────────┬───────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                SYNTHESIS & REASONING LAYER                       │
│  - Cross-validate signals                                        │
│  - Resolve conflicts                                             │
│  - Generate forecast narrative                                   │
│  - Structure JSON response                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    JSON RESPONSE (Schema)                        │
│  {                                                               │
│    "forecast_summary": "...",                                   │
│    "key_financial_trends": [...],                               │
│    "management_outlook": "...",                                 │
│    "risks_and_opportunities": [...],                            │
│    "source_documents": [...],                                   │
│    "errors": []                                                  │
│  }                                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
             ┌────────────┐   ┌─────────────┐
             │   CLIENT   │   │   MYSQL LOG │
             │  RESPONSE  │   │  (Async)    │
             └────────────┘   └─────────────┘
```

---

## 🛠️ Core Components Explained

### 1. FastAPI Application (`app/main.py`)
- **Lifespan management**: Initialize services on startup, cleanup on shutdown
- **Async endpoints**: Non-blocking request handling
- **Dependency injection**: Clean service wiring
- **Error handling**: Graceful degradation on failures
- **Background logging**: Async MySQL writes don't block responses

### 2. LangChain Agent (`app/services/agent.py`)
- **ReAct pattern**: Reason → Act → Observe loop
- **Tool orchestration**: Dynamically calls extraction and analysis tools
- **Error recovery**: Retries with exponential backoff
- **Prompt engineering**: Temperature control for deterministic extraction

### 3. Financial Data Extractor Tool
- **Purpose**: Extract metrics from quarterly reports
- **Method**: LLM-powered document parsing with source verification
- **Output**: Structured metrics with confidence scores
- **Guardrail**: All extracted values verified in source text

### 4. Qualitative Analysis Tool (RAG)
- **Purpose**: Analyze earnings calls for management sentiment
- **Method**: Vector search + LLM analysis of relevant transcript segments
- **Output**: Themes, sentiment, quotes, risks identified
- **Process**:
  1. Chunk earnings transcripts
  2. Generate embeddings
  3. Store in Pinecone
  4. Search on query
  5. Analyze retrieved context

### 5. Async MySQL Logger
- **Purpose**: Audit trail of all requests and responses
- **Implementation**: Non-blocking writes using `aiomysql`
- **Schema**: Captures request payload, response, errors, processing time
- **Benefit**: Complete reproducibility and governance

---

## 📊 AI Stack Components

| Layer | Technology | Why Chosen |
|-------|-----------|-----------|
| **LLM** | Claude 3.5 Sonnet | Excellent financial reasoning, affordable, reliable |
| **Embeddings** | text-embedding-3-small | Fast, cost-effective, good financial domain |
| **Vector DB** | Pinecone | Enterprise-ready, no ops needed |
| **PDF Parser** | Marker | Handles tables, preserves layout |
| **Framework** | LangChain | Best tool ecosystem, mature |
| **Web Server** | FastAPI | Async-first, auto-docs, fast |
| **Database** | MySQL 8.0 | Per requirements, proven for auditing |
| **Async Driver** | asyncmy | Non-blocking, good performance |

---

## 🔒 Guardrails & Safety

### 1. Source Verification
```python
# Every extraction verified against source text
if metric["source_quote"] not in document_text:
    mark_as_hallucination()
    exclude_from_forecast()
```

### 2. Confidence Scoring
```python
# Each metric gets 0-1 confidence:
# 1.0 = Explicitly stated in bold/table
# 0.8 = Clearly in body text
# 0.6 = Requires interpretation
# <0.6 = Excluded or flagged
```

### 3. Consistency Checking
```python
# Validate mathematical relationships:
# Net Margin = Net Profit / Revenue (tolerance: ±2%)
# Segments sum = Total Revenue (tolerance: ±100 Cr)
```

### 4. Hallucination Detection
```python
# If LLM claims value not in document:
# 1. Extract again with stricter prompt
# 2. If still not found, exclude metric
# 3. Flag in response for user review
```

### 5. Error Recovery
```python
# Tool fails:
# 1. Retry with exponential backoff (1s, 2s, 4s)
# 2. If max retries: Attempt simplified analysis
# 3. If still failing: Return partial forecast + error flags
# 4. All failures logged to MySQL with context
```

---

## 📈 Key Metrics & Monitoring

### Forecast Quality Metrics
- **Source Fidelity**: % of claims verifiable in source
- **Consistency Score**: % of mathematical validations passed
- **Confidence Coverage**: % of forecast with confidence > 0.8
- **Error Rate**: % of requests with errors

### System Performance Metrics
- **Latency**: Forecast generation time (target: < 30s)
- **Throughput**: Requests per minute
- **Success Rate**: % of requests returning complete forecast
- **Tool Execution Time**: Breakdown by tool (extraction, RAG search)

### Business Metrics
- **Forecast Accuracy**: MAPE (Mean Absolute Percentage Error) vs actuals
- **Confidence Calibration**: Does predicted uncertainty match reality?
- **Direction Accuracy**: % of forecasts getting growth direction right

---

## 🚨 Common Issues & Solutions

### Issue 1: "PDF parsing fails or extracts garbage"
**Solution**: 
- Use Marker parser instead of PyPDF2 (handles complex layouts)
- Pre-process PDFs with ImageMagick if quality issues
- Implement OCR fallback for scanned documents

### Issue 2: "LLM extracts wrong numbers (hallucination)"
**Solution**:
- Implement source verification (shown above)
- Use temperature=0.0 for extraction (deterministic)
- Use structured output schemas (Pydantic models)
- Require direct quotes in extraction

### Issue 3: "Vector search returns irrelevant segments"
**Solution**:
- Re-embed documents with better chunking (keep paragraphs together)
- Add metadata filters (date, speaker, document type)
- Use hybrid search (semantic + keyword/BM25)
- Manually curate important transcripts first

### Issue 4: "API times out on first request"
**Solution**:
- Pre-warm services on startup (fetch dummy data)
- Cache frequently requested documents
- Implement request timeout + graceful degradation
- Run agent with lower `max_iterations` if needed

### Issue 5: "MySQL logging slows down responses"
**Solution**:
- Move logging to background tasks (already implemented)
- Use connection pooling with proper sizing
- Batch writes during high load
- Add database indexes on frequent queries

---

## 📚 File Structure

```
tcs-forecast-agent/
├── app/
│   ├── main.py              ← FastAPI app (see starter_implementation.py)
│   ├── config.py            ← Settings management
│   ├── models/
│   │   ├── forecast.py      ← Request/response models
│   │   └── financial.py     ← Financial metric models
│   ├── services/
│   │   ├── agent.py         ← LangChain agent
│   │   ├── tools/
│   │   │   ├── financial_extractor.py
│   │   │   ├── qualitative_analysis.py
│   │   │   └── market_data.py
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   └── logger.py        ← Async MySQL logger (see starter_implementation.py)
│   └── routes/
│       ├── forecast.py
│       └── health.py
├── scripts/
│   ├── init_vector_store.py
│   ├── download_documents.py
│   ├── ingest_documents.py
│   └── test_agent.py
├── database/
│   └── init.sql             ← MySQL schema
├── tests/
│   ├── test_extraction.py
│   ├── test_agent.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt         ← All dependencies (see requirements.txt)
├── README.md
├── Dockerfile
└── docker-compose.yml
```

---

## 🔄 Development Workflow

### Local Development
```bash
# Terminal 1: Start server with auto-reload
uvicorn app.main:app --reload --port 8000

# Terminal 2: Run tests with watch
pytest tests/ -v --tb=short

# Terminal 3: Monitor logs
tail -f app.log
```

### Adding a New Document
```bash
# 1. Download PDF and place in data/financial_reports/
# 2. Run ingestion
python scripts/ingest_documents.py

# 3. Verify in Pinecone dashboard
# 4. Test with forecast endpoint
```

### Debugging an Issue
```bash
# 1. Check logs in app.log
# 2. Query MySQL for recent requests
SELECT * FROM api_request_logs 
WHERE status = 'error' 
ORDER BY timestamp DESC 
LIMIT 10;

# 3. Inspect full request/response
SELECT 
  request_payload,
  response_payload,
  error_messages,
  processing_time_ms
FROM api_request_logs
WHERE id = <request_id>;

# 4. Reproduce with test data
python scripts/test_agent.py
```

---

## 📦 Deployment Options

### Option 1: Local Machine (Development)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Container
```bash
docker build -t tcs-forecast:latest .
docker run -p 8000:8000 --env-file .env tcs-forecast:latest
```

### Option 3: Docker Compose (with MySQL)
```bash
docker-compose up -d
# Services: FastAPI, MySQL, optional Pinecone integration
```

### Option 4: Cloud Platform
- **AWS**: ECS + RDS MySQL + managed Pinecone
- **GCP**: Cloud Run + Cloud SQL + Vertex AI (for Gemini)
- **Azure**: App Service + Azure Database + Pinecone

---

## ✅ Validation Checklist Before Production

- [ ] All dependencies pinned in requirements.txt
- [ ] .env.example created (no secrets committed)
- [ ] Database schema executed and tables verified
- [ ] Vector store initialized and documents ingested
- [ ] All 5 tests passing (extraction, RAG, agent, API, integration)
- [ ] Health endpoint returning 200
- [ ] Forecast endpoint working with sample company
- [ ] MySQL logging confirmed (check DB)
- [ ] Error scenarios handled (missing docs, API timeouts)
- [ ] Response schema validated (matches required JSON)
- [ ] Performance acceptable (< 30s latency)
- [ ] Async logging non-blocking (responses fast even with DB lag)
- [ ] Source verification working (quotes match)
- [ ] Confidence scores populated
- [ ] Documentation complete

---

## 🎯 What's Next

1. **Customize master prompts** for your specific analysis needs
2. **Test with real TCS documents** (Q3 FY25 data)
3. **Tune hyperparameters** (temperature, embedding model, chunk size)
4. **Add domain-specific validation** (e.g., check EPS matches expectations)
5. **Implement forecast accuracy tracking** (compare to actual results)
6. **Build dashboard** for forecast monitoring
7. **Integrate with BI tools** (Tableau, PowerBI)
8. **Extend to other companies** (modify prompts, add company switching)
9. **Add user authentication** (for multi-tenant scenarios)
10. **Implement caching** (cache forecasts, documents, embeddings)

---

## 📞 Support & Debugging

### Common Commands

```bash
# Check if service is running
curl http://localhost:8000/health

# View recent requests
mysql -e "SELECT * FROM tcs_forecast.api_request_logs LIMIT 10;"

# Verify Pinecone setup
python -c "from pinecone import Pinecone; print(Pinecone().list_indexes())"

# Test financial extraction
python scripts/test_agent.py --test-extraction

# Rebuild vector store (dangerous - deletes old data)
python scripts/ingest_documents.py --rebuild
```

### Getting Help

1. Check `app.log` for error messages
2. Query MySQL logs for request context
3. Verify API credentials in .env
4. Check Pinecone dashboard for index status
5. Review LangChain debugging output (set verbose=True)

---

## 📝 License & Attribution

This guide implements best practices from:
- LangChain documentation and examples
- FastAPI production patterns
- Financial NLP literature
- Real-world RAG implementations

Feel free to adapt, extend, and deploy!

---

**Good luck building your AI forecasting system! 🚀**
