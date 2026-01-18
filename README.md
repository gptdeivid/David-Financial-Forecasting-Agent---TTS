# TCS Financial Forecasting Agent

An AI-powered financial analysis system that automatically generates business outlook forecasts for Tata Consultancy Services (TCS) by analyzing quarterly financial reports and earnings call transcripts.

## 🎯 Overview

This project implements an **autonomous AI agent** that:

- 📊 **Extracts quantitative metrics** from financial reports (revenue, margins, growth rates)
- 💬 **Analyzes qualitative signals** from earnings call transcripts (management sentiment, themes, risks)
- 🤖 **Synthesizes forecasts** by reasoning across multiple data sources
- 📝 **Generates structured JSON outputs** with complete source citations
- 🔍 **Maintains audit trail** via MySQL logging for governance and reproducibility

**Built with**: FastAPI • LangChain • Google Gemini / Claude • Chroma / Pinecone • MySQL

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- MySQL 8.0+
- API keys for:
  - Google Gemini API (or OpenAI/Claude)
  - OpenAI API (for embeddings)
  - Pinecone (for production vector store, optional)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd tcs-forecast-agent

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# 5. Initialize MySQL database
mysql -u root -p < database/init.sql

# 6. Initialize vector store and ingest documents
python scripts/init_vector_store.py
python scripts/download_documents.py
python scripts/ingest_documents.py

# 7. Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```

### Test the API

```bash
# Generate a forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"company": "TCS"}'

# Check API documentation
open http://localhost:8000/docs
```

---

## 📋 Project Structure

```
tcs-forecast-agent/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration management
│   ├── models/                    # Pydantic schemas
│   │   ├── forecast.py            # Request/response models
│   │   └── financial.py           # Financial metric models
│   ├── services/
│   │   ├── agent.py               # LangChain ReAct agent
│   │   ├── document_processor.py  # PDF parsing and processing
│   │   ├── vector_store.py        # RAG vector database
│   │   ├── logger.py              # Async MySQL logger
│   │   └── tools/
│   │       ├── financial_extractor.py      # Quantitative analysis
│   │       ├── qualitative_analysis.py     # Sentiment analysis
│   │       └── market_data.py              # Market data (optional)
│   └── routes/
│       ├── forecast.py            # Forecast endpoint
│       └── health.py              # Health check
├── scripts/
│   ├── download_documents.py      # Fetch TCS financial documents
│   ├── ingest_documents.py        # Populate vector store
│   ├── init_vector_store.py       # Initialize Chroma/Pinecone
│   └── test_agent.py              # Manual testing utilities
├── database/
│   └── init.sql                   # MySQL schema
├── tests/
│   ├── test_extraction.py         # Unit tests for extraction
│   ├── test_qualitative.py        # Unit tests for RAG
│   ├── test_agent.py              # Integration tests
│   └── test_api.py                # API endpoint tests
├── data/
│   ├── financial_reports/         # Downloaded PDFs
│   └── earnings_transcripts/      # Earnings call transcripts
├── .env.example                   # Environment variables template
├── .gitignore
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container definition
├── docker-compose.yml             # Multi-container setup
├── README.md                      # This file
├── IMPLEMENTATION_PLAN.md         # Detailed implementation guide
└── QUICK_START.md                 # Quick reference guide
```

---

## 🏗️ Architecture

### High-Level Flow

```
User Request
    ↓
FastAPI Endpoint
    ↓
LangChain ReAct Agent
    ↓
┌─────────────────┬──────────────────┬──────────────┐
│                 │                  │              │
Financial         Qualitative        Market Data
Extractor         Analysis           (Optional)
(LLM + PDF)       (RAG + LLM)        (API)
│                 │                  │
└─────────────────┴──────────────────┴──────────────┘
    ↓
Synthesis & Reasoning
    ↓
Structured JSON Response + MySQL Audit Log
```

### Key Components

1. **Document Acquisition**: Web scraping and PDF downloading from Screener.in and TCS investor relations
2. **Document Processing**: PDF parsing with Gemini Vision API or Marker for complex layouts
3. **Vector Store (RAG)**: Semantic search across earnings call transcripts using Chroma/Pinecone
4. **LLM Tools**:
   - **FinancialDataExtractorTool**: Extracts metrics from financial reports
   - **QualitativeAnalysisTool**: Analyzes management sentiment and themes
   - **MarketDataTool**: Fetches current market context (optional)
5. **Agent Orchestrator**: LangChain ReAct agent that coordinates tool execution
6. **API Layer**: FastAPI with async request handling and background logging
7. **Audit Logger**: MySQL database for complete request/response tracking

---

## 🤖 AI Reasoning Approach

### ReAct Pattern (Reason → Act → Observe)

The agent follows an iterative reasoning loop:

1. **REASON**: Analyze the forecast request and decide what information is needed
2. **ACT**: Execute specialized tools (financial extraction, qualitative analysis)
3. **OBSERVE**: Receive and integrate tool outputs
4. **SYNTHESIZE**: Generate forecast by cross-validating quantitative and qualitative signals

### Example Flow

```
User: "Generate Q4 FY25 forecast for TCS"
    ↓
Agent Thinks: "I need Q3 metrics + management sentiment"
    ↓
Agent Acts:
  - Extract metrics from Q3 FY25 report
  - Analyze earnings call for themes
    ↓
Agent Observes:
  - Revenue: +5.4% YoY, Margin: 21%
  - Sentiment: Cautious on client spending, positive on AI
    ↓
Agent Synthesizes:
  "Q4 growth moderates to 3-5% due to client caution,
   offset by AI services momentum"
```

### Guardrails

- **Source Verification**: Every extracted metric verified against source text
- **Confidence Scoring**: 0-1 scale, exclude claims with confidence < 0.6
- **Consistency Checks**: Mathematical validation (e.g., margin = profit / revenue)
- **Error Recovery**: Automatic retry with exponential backoff

---

## 📊 Sample Output

```json
{
  "forecast_summary": "TCS expects moderate revenue growth of 3-5% in Q4 FY25...",
  "key_financial_trends": [
    "Q3 FY25 Revenue: ₹62,613 Cr (+5.4% YoY)",
    "Operating Margin: 21% stable",
    "AI Services: Strong demand momentum"
  ],
  "management_outlook": "Cautiously optimistic. Management acknowledged client spending pause...",
  "risks_and_opportunities": {
    "risks": ["Client spending pause may persist", "BFSI caution"],
    "opportunities": ["AI services momentum", "Digital transformation"]
  },
  "source_documents": [
    "TCS Q3 FY25 Results - Investor Release",
    "TCS Q3 FY25 Earnings Call Transcript"
  ],
  "reasoning": {
    "quantitative_signal": "5.4% YoY growth with stable margins...",
    "qualitative_signal": "Management cautious on near-term...",
    "confidence_level": "Medium-High"
  },
  "errors": []
}
```

---

## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini 1.5 Pro / Claude 3.5 Sonnet | Document analysis, reasoning, synthesis |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic search |
| **Vector DB** | Chroma (dev) / Pinecone (prod) | RAG for earnings calls |
| **PDF Parser** | Gemini Vision API / Marker | Extract data from complex tables |
| **Framework** | LangChain | Agent orchestration, tool management |
| **Web Server** | FastAPI | Async API endpoints |
| **Database** | MySQL 8.0 | Audit logging |
| **Web Scraping** | BeautifulSoup / Selenium | Document acquisition |

---

## 📚 Documentation

- **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)**: Comprehensive 10-section implementation guide with architecture, development phases, technical requirements, and deployment strategies
- **[QUICK_START.md](./QUICK_START.md)**: Quick reference guide with setup instructions and troubleshooting
- **API Documentation**: Auto-generated at `http://localhost:8000/docs` when server is running

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_extraction.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 🚢 Deployment

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f app
```

### Cloud Deployment

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md#8-deployment-strategy) for detailed cloud deployment instructions (AWS, GCP, Azure).

---

## 📈 Success Metrics

### Technical Metrics

- **API Latency**: < 30s (p95)
- **Success Rate**: > 95%
- **Hallucination Rate**: < 5%
- **Source Fidelity**: > 90%

### Business Metrics

- **Forecast Accuracy (Revenue MAPE)**: < 10%
- **Direction Accuracy**: > 70%
- **Time Savings**: 80% vs manual analysis

---

## 🛠️ Development

### Adding New Documents

```bash
# 1. Place PDF in data/financial_reports/
# 2. Run ingestion
python scripts/ingest_documents.py

# 3. Verify in vector store
# 4. Test with forecast endpoint
```

### Debugging

```bash
# Check application logs
tail -f app.log

# Query MySQL for recent requests
mysql -e "SELECT * FROM tcs_forecast_db.api_request_logs WHERE status='error' LIMIT 10;"

# Run agent in debug mode
python scripts/test_agent.py --verbose
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **LangChain**: For the excellent agent orchestration framework
- **Google Gemini**: For multimodal document understanding capabilities
- **FastAPI**: For the modern, async web framework
- **TCS**: For publicly available financial documents

---

## 📞 Support

For questions or issues:

1. Check the [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for detailed guidance
2. Review [QUICK_START.md](./QUICK_START.md) for common issues
3. Check application logs and MySQL audit trail
4. Open an issue on GitHub

---

## 🗺️ Roadmap

- [ ] Support for multiple companies (Infosys, Wipro, HCL)
- [ ] Real-time market data integration
- [ ] Forecast accuracy tracking dashboard
- [ ] Multi-quarter trend analysis
- [ ] Automated document ingestion pipeline
- [ ] Web UI for forecast visualization
- [ ] Export to PDF/PowerPoint reports

---

**Built with ❤️ for financial analysts and investors**
