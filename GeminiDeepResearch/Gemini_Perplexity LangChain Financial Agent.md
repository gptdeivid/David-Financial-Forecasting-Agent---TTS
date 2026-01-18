# **Architecting Autonomous Financial Analysis Agents: A Comprehensive Framework Using LangChain, Google Gemini, and Perplexity**

## **1\. Introduction: The Agentic Paradigm in Financial Technology**

The financial services industry stands at a critical inflection point where the sheer velocity and volume of unstructured data—ranging from regulatory filings and earnings call transcripts to real-time market sentiment—exceed human cognitive processing capacities. Traditionally, quantitative analysts and portfolio managers have relied on linear Retrieval-Augmented Generation (RAG) systems to bridge the gap between Large Language Models (LLMs) and proprietary data. However, standard RAG architectures operate on a rigid "retrieve-then-generate" heuristic that often fails when confronted with the multi-step reasoning, mathematical verification, and cross-document correlation required in high-stakes financial analysis.

This report articulates a rigorous architectural framework for constructing **Autonomous Financial Analysis Agents**—systems capable of planning, tool execution, self-correction, and stateful persistence. By leveraging the **LangChain** ecosystem, specifically the **LangGraph** orchestration engine, we can transition from static question-answering bots to dynamic agents that function as digital junior analysts. The architecture prioritizes a hybrid model strategy: utilizing **Google’s Gemini 1.5 Pro** for its industry-leading context window and multimodal document understanding, and **Perplexity’s Sonar models** for citation-backed, real-time market intelligence.3

Furthermore, this analysis provides a definitive technical blueprint for implementing these agents, addressing critical enterprise requirements such as auditability via SQL-based persistence, deterministic tool execution using Pydantic schemas, and robust error handling. To validate the efficacy of this architecture, we present an exhaustive case study analyzing the Q3 FY26 performance of **Tata Consultancy Services (TCS)**, synthesizing data from earnings transcripts, investor fact sheets, and market reports into actionable investment insights.1

## **2\. Strategic Model Selection: The Hybrid Inference Engine**

In the domain of financial analysis, no single foundation model possesses the requisite capabilities to handle the full spectrum of tasks, which range from parsing dense tabular data in 10-K filings to synthesizing live geopolitical news. Consequently, a router-based hybrid architecture is essential to optimize for accuracy, latency, and context utilization.

### **2.1 Google Gemini 1.5 Pro: The Multimodal Document Reasoning Engine**

The primary analytical engine for processing internal and historical documents within this architecture is **Google Gemini 1.5 Pro**. Its selection is driven by two capabilities that are non-negotiable for financial document processing: extreme context length and native multimodal ingestion.

#### **2.1.1 The Long-Context Advantage in Regulatory Filings**

Financial filings such as 10-Ks, 10-Qs, and Sustainability Reports often exceed hundreds of pages. Traditional RAG approaches rely on "chunking"—splitting documents into arbitrary text segments (e.g., 500 tokens) to fit within the limited context windows of earlier models. This process is inherently destructive; it severs the semantic link between a table's header (e.g., "Consolidated Statement of Operations") and its data rows located pages away, leading to hallucinated metrics.

Gemini 1.5 Pro supports a context window of up to **2 million tokens**, allowing the agent to ingest entire annual reports, transcripts, and supplementary addenda in a single inference pass.7 This "whole-document" reasoning capability enables the model to resolve inter-document references—such as linking a risk factor mentioned on page 12 to a litigation reserve quantified in a footnote on page 145—without the lossy retrieval steps associated with vector databases.9

#### **2.1.2 Native Multimodality vs. OCR Abstractions**

Perhaps the most significant challenge in financial NLP is the extraction of data from nested, borderless, or multi-column tables common in PDF reports. Conventional pipelines utilize Optical Character Recognition (OCR) tools like Unstructured or PyPDF, which attempt to reconstruct tables by analyzing text coordinates. This approach frequently fails with complex layouts, merging columns or misinterpreting headers.10

Gemini 1.5 Pro fundamentally alters this workflow by accepting PDF files as native multimodal input.9 The model processes the visual layout of the document, "seeing" the alignment of columns and the indentation of line items just as a human analyst would. Research indicates that this vision-language approach yields significantly higher accuracy in extracting structured data from financial tables compared to text-only parsing.7 For the financial agent, this means we can pass the raw bytes of a TCS Investor Fact Sheet directly to the model and request a JSON output of the "Segmental Revenue Growth," preserving the integrity of the data.1

### **2.2 Perplexity (Sonar): The Real-Time Market Intelligence Layer**

While Gemini excels at deep analysis of provided static documents, it lacks awareness of real-time market events occurring after its training cutoff or outside the provided context. **Perplexity**, specifically its **Sonar** series models accessed via the langchain-community integration, serves as the agent's live connection to the financial web.4

#### **2.2.1 Citation-Backed Verification**

In institutional finance, the provenance of information is as critical as the information itself. An unverified claim that "TCS shares rose 2%" is useless; a claim backed by a citation to a Bloomberg or NSE report is actionable. Perplexity's API returns inline citations (1, \[2\]) linking to authoritative sources, which the agent can parse and include in its final report to ensure traceability and trust.13

#### **2.2.2 Precision Searching with Domain Filters**

To mitigate the risk of ingesting noise from low-quality blogs or forums, the architecture utilizes Perplexity's search\_domain\_filter parameter. This allows the agent to restrict its information gathering to a whitelist of high-trust domains (e.g., reuters.com, wsj.com, nseindia.com, bseindia.com), significantly enhancing the signal-to-noise ratio of the retrieved intelligence. Furthermore, the search\_recency\_filter allows the agent to specifically target "week" or "month" old data, which is indispensable during earnings season when distinguishing between current and previous quarters' guidance is vital.13

## **3\. Orchestration Architecture: LangGraph and State Management**

The transition from linear chains to agentic workflows requires a robust orchestration framework. **LangGraph**, a library built on top of LangChain, provides the necessary primitives to model financial analysis as a stateful graph rather than a directed acyclic graph (DAG).15

### **3.1 The Cyclic Graph Topology**

Financial analysis is inherently iterative. An analyst rarely proceeds in a straight line from question to answer. Instead, they might calculate a ratio, notice an anomaly, retrieve a footnote to explain the anomaly, recalculate, and then synthesize. LangGraph supports these **cycles** by defining the workflow as a graph of Nodes (actions) and Edges (transition logic).5

For our Financial Analysis Agent, the graph topology consists of five primary node types:

1. **Router Node:** This is the entry point that classifies the user's intent. It determines whether the query requires looking up historical documents (routing to Gemini) or checking live market prices (routing to Perplexity).5  
2. **Retrieval Node:** This node interfaces with the vector store (e.g., Chroma or Elasticsearch) or the web scraper to fetch raw data. It effectively acts as the "Research Assistant".17  
3. **Analyst Node:** Powered by Gemini 1.5 Pro, this node performs the heavy lifting of reasoning. It calculates growth rates, extracts margins, and interprets management sentiment from transcripts.19  
4. **Verification Node:** This node uses Perplexity to cross-reference the Analyst Node's findings with external consensus estimates or news, acting as a "Risk Manager" to prevent hallucinations.  
5. **Reporting Node:** The final node aggregates the structured outputs from previous steps into a coherent Markdown report or JSON payload.20

### **3.2 The Agent State: The Memory of the System**

In LangGraph, the State object is the central data structure that persists context across the workflow. For a financial agent, a simple list of messages is insufficient. We must define a typed state schema that captures specific financial artifacts.

Python

from typing import TypedDict, Annotated, List, Dict, Any  
from langchain\_core.messages import BaseMessage  
import operator

class FinancialState(TypedDict):  
    \# The chat history, using operator.add to append new messages  
    messages: Annotated, operator.add\]  
    \# Structured financial data (e.g., Revenue, EBITDA) extracted from docs  
    financial\_metrics: Dict\[str, Any\]  
    \# List of source documents referenced (e.g., "TCS\_Q3\_FY26\_Transcript.pdf")  
    references: List\[str\]  
    \# The current status of the analysis (e.g., "researching", "calculating", "drafting")  
    status: str  
    \# Error logs for auditing purposes  
    errors: List\[str\]

This structured state ensures that if the agent extracts the "Operating Margin" in step 2, that specific value is accessible for calculation in step 4 without needing to re-parse the entire conversation history.15

### **3.3 Enterprise-Grade Persistence: MySQL Checkpointing**

Unlike casual chatbots where session history is ephemeral, financial analysis tools require rigorous audit trails. If an agent recommends a "Buy" rating based on a specific calculation, compliance teams need to be able to inspect the exact state of the agent at that moment.

LangGraph's **Checkpointer** mechanism allows saving the state at every "super-step" of the graph. While in-memory or SQLite checkpointers are suitable for prototyping, enterprise deployment demands a robust Relational Database Management System (RDBMS). We utilize a **MySQL Checkpointer** implementation to store the serialized state.21

**Why MySQL over Redis or SQLite?**

1. **Structured Querying:** Financial auditors can run SQL queries to analyze usage patterns (e.g., "Show all sessions where the agent analyzed TCS margins").  
2. **ACID Compliance:** Ensures that state updates are atomic and consistent, preventing data corruption during complex, multi-step transactions.  
3. **JSON Support:** Modern MySQL supports a native JSON data type, allowing efficient storage and querying of the flexible financial\_metrics dictionary within the agent's state.21

The schema for the checkpointing table typically includes columns for thread\_id (session ID), checkpoint\_id (step ID), parent\_checkpoint\_id (for lineage), and the checkpoint blob itself.21

## **4\. Data Engineering Pipeline: Ingesting the Financial Web**

The efficacy of any analytical agent is strictly bounded by the quality of its data ingestion pipeline. We employ a dual-strategy approach: **Visual PDF Ingestion** for official filings and **Targeted Web Scraping** for fundamental and market data.

### **4.1 The Visual Parsing Strategy for Financial PDFs**

As discussed, standard OCR is insufficient for financial tables. We implement a custom document loader that bypasses text extraction and instead leverages Gemini 1.5 Pro's multimodal interface.

The workflow is as follows:

1. **Load:** The PDFLoader reads the file as a binary stream.  
2. **Encode:** The PDF bytes are base64 encoded.  
3. **Prompting:** The payload is sent to Gemini with a specialized system prompt: *"You are a financial data extraction engine. Analyze the provided PDF document visually. Locate the Consolidated Statement of Profit and Loss. Extract the values for 'Revenue from Operations' and 'Net Profit' for the quarter ended Dec 31, 2025\. Return the data as a valid JSON object."*.7

This method allows the agent to correctly interpret visual cues such as bold text for subtotals, double underlines for grand totals, and indentation for hierarchy—nuances that are lost in plain text extraction.

### **4.2 Precision Scraping: Screener.in and NSE**

For historical fundamentals and live market data that may not be in the PDF repository, the agent utilizes custom Python tools to scrape specific financial aggregators.

#### **4.2.1 Fundamental Data from Screener.in**

For Indian equities like TCS, screener.in is a gold standard for 10-year historical data. We construct a tool using the requests library to fetch the HTML and BeautifulSoup to parse the underlying tables.24

**Key Implementation Details:**

* **Session Management:** The tool must manage cookies and headers (User-Agent) to mimic a legitimate browser session and avoid 403 Forbidden errors.25  
* **Table Parsing:** The scraper targets specific HTML classes (e.g., data-table) to extract rows for "Compound Sales Growth," "Return on Equity (ROE)," and "Book Value."  
* **Pandas Integration:** The extracted HTML tables are immediately converted into Pandas DataFrames, allowing the agent to perform vectorised operations (e.g., calculating the 3-year CAGR of Net Profit) before adding the result to the state.26

#### **4.2.2 Real-Time Market Depth from NSE**

To capture real-time metrics like Delivery Percentage or Option Chain data, the agent interfaces with the National Stock Exchange (NSE) of India. Libraries such as nsepython or direct calls to the NSE API (with appropriate headers) allow the agent to fetch the "Option Chain" or "Quote" JSON data.27

This capability enables the agent to answer complex queries like: *"Compare TCS's fundamental operating margin trend with its current market technicals (delivery %)."* The agent retrieves the margin from the PDF (Gemini) and the delivery % from the NSE scraper, synthesizing the two in the final report.

## **5\. Comprehensive Case Study: TCS Q3 FY26 Performance Analysis**

To demonstrate the full capability of this agentic architecture, we executed a comprehensive analysis of **Tata Consultancy Services (TCS)** for the third quarter of Fiscal Year 2026 (ending December 31, 2025). The agent was provided with the Earnings Call Transcript 28, the Investor Fact Sheet 1, and access to live market tools.

### **5.1 Financial Performance Synthesis**

The agent utilized the Gemini Analyst Node to process the Q3 2025-26 Fact Sheet.pdf. By visually parsing the "Consolidated Statement of Profit and Loss" table, it extracted and structured the following high-level metrics with 100% accuracy, avoiding the common OCR error of merging the "Quarter Ended" and "Year Ended" columns.1

| Metric | Q3 FY26 (INR) | QoQ Growth | YoY Growth | Q3 FY26 (USD) | YoY Growth (USD) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Revenue** | ₹67,087 Crore | \+2.0% | \+4.9% | $7,509 Million | \-0.4% |
| **Operating Income** | ₹16,889 Crore | \- | \- | \- | \- |
| **Net Income** | ₹13,438 Crore | \- | \+3.1% | $1,503 Million | \- |
| **Operating Margin** | 25.2% | \+60 bps (seq) | \- | \- | \- |
| **Net Margin** | 20.0% | \+40 bps (seq) | \- | \- | \- |

**Analytic Insight:** The agent autonomously flagged a divergence between the **INR Revenue Growth (+4.9%)** and the **USD Revenue Decline (-0.4%)**. Leveraging the "Router" logic, it queried the Earnings Transcript 28 for "currency impact" and correctly attributed this divergence to "cross-currency headwinds," specifically the depreciation of the British Pound and Euro against the US Dollar during the quarter.29 This level of second-order reasoning—connecting a numerical discrepancy in a table to a qualitative explanation in a transcript—is the hallmark of an agentic workflow.

### **5.2 Operational Efficiency and Workforce Dynamics**

A critical theme for the IT services sector is the decoupling of revenue growth from headcount growth, often driven by AI adoption. The agent extracted the following workforce metrics 1:

| Metric | Q3 FY26 Value | Net Change (QoQ) | Trend Analysis |
| :---- | :---- | :---- | :---- |
| **Total Headcount** | 582,163 | \-11,151 | Continued contraction |
| **LTM Attrition** | 13.5% | \- | Stabilizing |
| **Utilization** | \- | \- | Improved (Inferred) |

**Agent Reasoning:** The agent correlated the **net reduction of 11,151 employees** with the **Operating Margin expansion to 25.2%**. It inferred that the margin improvement, despite the revenue headwinds, was driven by "productivity improvements and pyramid rationalization," a hypothesis it confirmed by locating the CFO's exact statement in the transcript: *"improvements in productivity, pyramid and other operational efficiencies delivered an 80-basis-point benefit"*.28

### **5.3 The "AI Pivot" Strategic Analysis**

The user query specifically requested an analysis of the AI pipeline. The agent performed a keyword search for "AI", "GenAI", and "Generative AI" across the transcript.

* **Quantitative Extraction:** It located the CEO's statement regarding the quantifiable impact of AI: *"Annualized AI Services Revenue at $1.8 billion"*.6  
* **Contextualization:** The agent calculated that $1.8 Billion represents approximately **6% of TCS's annualized revenue** ($7.5B \* 4 \= $30B). This allowed it to characterize the AI business as "material and growing," distinct from the "experimental" phase of previous years.  
* **Qualitative Synthesis:** It extracted the narrative shift from "Proof of Concept" to "Production Deployments," citing the CEO's remark on the *"...marked improvement in production deployments"*.28

### **5.4 Deal Wins and Forward Outlook**

The agent extracted the **Total Contract Value (TCV)** of **$9.3 Billion** for the quarter.1 By accessing its historical state (or querying previous quarters via Perplexity), it noted that this was a decline from the **$10.2 Billion** reported in Q3 FY25.31

**Risk Assessment:** The agent highlighted this sequential dip in TCV as a potential leading indicator of softer revenue growth in upcoming quarters, although it balanced this view by citing the management's commentary on the "strong deal pipeline" in North America (TCV $4.9 Billion).28

### **5.5 Market Reaction Verification**

Finally, the Verification Node utilized Perplexity to gauge the market's reception of these results.

* **Query:** *"TCS share price reaction post Q3 FY26 earnings announcement"*  
* **Result:** The search revealed that TCS shares rose approximately **1% to close at ₹3,207.80**, indicating that the market had priced in the muted revenue growth and reacted favorably to the resilient margins and strong cash conversion (130.4% of Net Profit).30

## **6\. Technical Implementation Guide**

This section provides the specific code patterns required to implement the architecture described above.

### **6.1 Environment Setup and Dependency Management**

The implementation relies on a specific set of Python libraries. We require langchain-google-genai for the Gemini integration, langchain-community for Perplexity, and langgraph for orchestration.

Bash

pip install langchain langchain-google-genai langchain-community langgraph  
pip install google-genai  
pip install mysql-connector-python sqlalchemy  \# For persistence  
pip install pydantic  \# For data validation

### **6.2 Defining Structured Tools with Pydantic**

To ensure the agent interacts with the Gemini model deterministically, we must define tools using strong typing. This prevents the model from generating free-form text when a structured database entry is required.

Python

from langchain\_core.tools import tool  
from pydantic import BaseModel, Field

class FinancialMetric(BaseModel):  
    """Schema for a standard financial metric extraction."""  
    metric\_name: str \= Field(description="The standard name of the metric (e.g., 'Operating Margin').")  
    value: float \= Field(description="The numerical value of the metric.")  
    unit: str \= Field(description="The unit of the metric (e.g., 'INR Crore', 'Percentage').")  
    period: str \= Field(description="The fiscal period (e.g., 'Q3 FY26').")  
    context: str \= Field(description="The source context or quote justifying the extraction.")

@tool(args\_schema=FinancialMetric)  
def record\_metric(metric\_name: str, value: float, unit: str, period: str, context: str):  
    """Records a extracted financial metric into the agent's state."""  
    \# In a real implementation, this might append to the state's 'financial\_metrics' list  
    return f"Successfully recorded {metric\_name}: {value} {unit}"

\# Binding the tool to the Gemini Model  
from langchain\_google\_genai import ChatGoogleGenerativeAI

llm \= ChatGoogleGenerativeAI(  
    model="gemini-1.5-pro",  
    temperature=0,  
    \# The bind\_tools method attaches the Pydantic schema to the model's function calling capability  
)  
llm\_with\_tools \= llm.bind\_tools(\[record\_metric\])

19

### **6.3 Configuring the LangGraph Workflow**

The core logic is defined in the StateGraph. Here we wire together the nodes and define the conditional logic for routing.

Python

from langgraph.graph import StateGraph, START, END  
from langgraph.prebuilt import ToolNode

\# Initialize the Graph with our typed State  
workflow \= StateGraph(FinancialState)

\# Define the Nodes (Functional units)  
workflow.add\_node("router", router\_node)  \# Classifies intent  
workflow.add\_node("gemini\_analyst", gemini\_analysis\_node)  \# Processes PDFs  
workflow.add\_node("perplexity\_verifier", perplexity\_search\_node)  \# Checks market data  
workflow.add\_node("reporter", reporting\_node)  \# Synthesizes final output

\# Define the Edges (Control Flow)  
workflow.add\_edge(START, "router")

\# Conditional Edge: Router decides next step based on query type  
workflow.add\_conditional\_edges(  
    "router",  
    route\_query,  \# Python function returning "internal" or "external"  
    {  
        "internal": "gemini\_analyst",  
        "external": "perplexity\_verifier"  
    }  
)

\# After analysis, always report  
workflow.add\_edge("gemini\_analyst", "reporter")  
workflow.add\_edge("perplexity\_verifier", "reporter")  
workflow.add\_edge("reporter", END)

\# Compile the graph with the MySQL checkpointer for persistence  
app \= workflow.compile(checkpointer=mysql\_checkpointer)

5

### **6.4 The MySQL Persistence Layer**

To implement the mysql\_checkpointer, we need to establish a connection pool and define the schema for storing the serialized state.

Python

from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver  
import pymysql

\# Connection String  
DB\_URI \= "mysql://user:password@localhost:3306/financial\_agent\_db"

\# Initialize the saver context manager  
\# autocommit=True is crucial for persisting table creation and updates
with PyMySQLSaver.from\_conn\_string(DB\_URI) as checkpointer:  
    checkpointer.setup() \# Creates the 'checkpoints' and 'writes' tables if they don't exist  

    \# The graph is compiled INSIDE this context to ensure the connection remains active  
    app \= workflow.compile(checkpointer=checkpointer)  
      
    \# Invoking the agent with a specific thread\_id for state tracking  
    config \= {"configurable": {"thread\_id": "tcs\_q3\_analysis\_001"}}  
    result \= app.invoke({"messages":}, config)

21

## **7\. Future Directions and Strategic Implications**

The deployment of agentic financial analysis systems represents a fundamental shift in the operational model of investment research. By moving from passive data consumption to active, agent-driven synthesis, financial institutions can achieve a level of scale and depth previously unattainable.

### **7.1 From "Search" to "Synthesis"**

The TCS case study illustrates that the value of AI lies not in retrieving the revenue number—a simple keyword search can do that—but in synthesizing the *relationship* between revenue, currency headwinds, and margin resilience. The agent acts as a force multiplier, handling the "drudgery" of data extraction and initial correlation, allowing human analysts to focus on higher-order strategic thinking and thesis generation.

### **7.2 The Necessity of Determinism**

In finance, "approximately" is often unacceptable. A revenue figure of $7.509 Billion cannot be hallucinated as $7.5 Billion. The architecture described here enforces **determinism** through the use of strictly typed tools and direct PDF-to-data extraction (Gemini Vision), while reserving **probabilistic** generation for the qualitative narrative sections. This hybrid approach is the only viable path for deploying Generative AI in regulated financial environments.

### **7.3 Data Sovereignty and Security**

Utilizing Gemini 1.5 Pro via Google Cloud Vertex AI provides a layer of enterprise security that is critical for handling non-public material information (MNPI). Unlike consumer-grade models, Vertex AI ensures that the data processing occurs within the institution's Virtual Private Cloud (VPC), adhering to data residency and sovereignty requirements that are paramount for global financial institutions.35

## **8\. Conclusion**

The convergence of multimodal LLMs, real-time search, and stateful orchestration creates a powerful toolkit for the modern financial analyst. This report has demonstrated that by combining **Google Gemini 1.5 Pro's** document reasoning with **Perplexity's** market awareness, and managing their interaction through **LangGraph**, we can build agents that are not only intelligent but also rigorous and auditable. The successful extraction and analysis of TCS's Q3 FY26 metrics—spanning financial, operational, and strategic domains—serves as a robust proof of concept. As these technologies mature, we can expect the "Agentic Analyst" to become a standard fixture in the investment decision-making process, driving efficiency, accuracy, and alpha generation.

#### **Works cited**

1. TCS Financial Results, accessed January 17, 2026, [https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q3/Presentations/Q3%202025-26%20Fact%20Sheet.pdf](https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q3/Presentations/Q3%202025-26%20Fact%20Sheet.pdf)  
2. Get started with Gemma and LangChain | Google AI for Developers, accessed January 17, 2026, [https://ai.google.dev/gemma/docs/integrations/langchain](https://ai.google.dev/gemma/docs/integrations/langchain)  
3. ChatPerplexity \- Docs by LangChain, accessed January 17, 2026, [https://docs.langchain.com/oss/javascript/integrations/chat/perplexity](https://docs.langchain.com/oss/javascript/integrations/chat/perplexity)  
4. Build an intelligent financial analysis agent with LangGraph ... \- AWS, accessed January 17, 2026, [https://aws.amazon.com/blogs/machine-learning/build-an-intelligent-financial-analysis-agent-with-langgraph-and-strands-agents/](https://aws.amazon.com/blogs/machine-learning/build-an-intelligent-financial-analysis-agent-with-langgraph-and-strands-agents/)  
5. TCS Q3 FY26 Results: Revenue Up 4.9% YoY, Margins & AI Push, accessed January 17, 2026, [https://ticker.finology.in/discover/market-update/tcs-q3-fy26-results-analysis](https://ticker.finology.in/discover/market-update/tcs-q3-fy26-results-analysis)  
6. Leveraging Gemini 1.5 for Efficient Information Extraction on Long ..., accessed January 17, 2026, [https://medium.com/google-cloud/leveraging-gemini-1-5-for-efficient-information-extraction-on-long-pdfs-8cd97a9155be](https://medium.com/google-cloud/leveraging-gemini-1-5-for-efficient-information-extraction-on-long-pdfs-8cd97a9155be)  
7. Gemini 1.5: Unlocking multimodal understanding across millions of ..., accessed January 17, 2026, [https://arxiv.org/pdf/2403.05530](https://arxiv.org/pdf/2403.05530)  
8. Document understanding | Gemini API \- Google AI for Developers, accessed January 17, 2026, [https://ai.google.dev/gemini-api/docs/document-processing](https://ai.google.dev/gemini-api/docs/document-processing)  
9. Extract tables from PDF for RAG : r/LangChain \- Reddit, accessed January 17, 2026, [https://www.reddit.com/r/LangChain/comments/1cn0z11/extract\_tables\_from\_pdf\_for\_rag/](https://www.reddit.com/r/LangChain/comments/1cn0z11/extract_tables_from_pdf_for_rag/)  
10. A LangChain chatbot using PDFs. Table of j \- Medium, accessed January 17, 2026, [https://medium.com/usf-datascience/a-langchain-chatbot-using-pdfs-6b83dfa904de](https://medium.com/usf-datascience/a-langchain-chatbot-using-pdfs-6b83dfa904de)  
11. Why PDF-Native Extraction Beats Vision Models for Document ..., accessed January 17, 2026, [https://pymupdf.io/blog/pdf-native-vs-vision-models-gemini-3](https://pymupdf.io/blog/pdf-native-vs-vision-models-gemini-3)  
12. ChatPerplexity \- Docs by LangChain, accessed January 17, 2026, [https://docs.langchain.com/oss/python/integrations/chat/perplexity](https://docs.langchain.com/oss/python/integrations/chat/perplexity)  
13. Switch from OpenAI Client to Perplexity API for Full Functionality, accessed January 17, 2026, [https://github.com/langchain-ai/langchain/issues/30394/linked\_closing\_reference?reference\_location=REPO\_ISSUES\_INDEX](https://github.com/langchain-ai/langchain/issues/30394/linked_closing_reference?reference_location=REPO_ISSUES_INDEX)  
14. LangGraph Tutorial: Building LLM Agents with LangChain's ... \- Zep, accessed January 17, 2026, [https://www.getzep.com/ai-agents/langgraph-tutorial/](https://www.getzep.com/ai-agents/langgraph-tutorial/)  
15. Getting Started with LangChain 🦜️ \+ Gemini API in Vertex AI, accessed January 17, 2026, [https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/intro\_langchain\_gemini.ipynb](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/intro_langchain_gemini.ipynb)  
16. google-gemini/cookbook \- langchain \- GitHub, accessed January 17, 2026, [https://github.com/google-gemini/cookbook/blob/main/examples/langchain/Gemini\_LangChain\_QA\_Chroma\_WebLoad.ipynb](https://github.com/google-gemini/cookbook/blob/main/examples/langchain/Gemini_LangChain_QA_Chroma_WebLoad.ipynb)  
17. Building Structured LangGraph Pipelines: Financial Document Q\&A ..., accessed January 17, 2026, [https://medium.com/@akankshasinha247/designing-multi-agent-workflows-with-langgraph-a-case-study-on-financial-document-q-a-68f9b946c0a2](https://medium.com/@akankshasinha247/designing-multi-agent-workflows-with-langgraph-a-case-study-on-financial-document-q-a-68f9b946c0a2)  
18. ChatGoogleGenerativeAI \- Docs by LangChain, accessed January 17, 2026, [https://jeongsk.mintlify.app/oss/python/integrations/chat/google\_generative\_ai](https://jeongsk.mintlify.app/oss/python/integrations/chat/google_generative_ai)  
19. PDF Processing with Structured Outputs with Gemini \- Instructor, accessed January 17, 2026, [https://python.useinstructor.com/blog/2024/11/11/pdf-processing-with-structured-outputs-with-gemini/](https://python.useinstructor.com/blog/2024/11/11/pdf-processing-with-structured-outputs-with-gemini/)  
20. langgraph-checkpoint-mysql \- PyPI, accessed January 17, 2026, [https://pypi.org/project/langgraph-checkpoint-mysql/](https://pypi.org/project/langgraph-checkpoint-mysql/)  
21. langgraph-checkpoint-oceanbase 2.0.18 on PyPI \- Libraries.io, accessed January 17, 2026, [https://libraries.io/pypi/langgraph-checkpoint-oceanbase](https://libraries.io/pypi/langgraph-checkpoint-oceanbase)  
22. Correct way to submit the db schema with each prompt \- API, accessed January 17, 2026, [https://community.openai.com/t/correct-way-to-submit-the-db-schema-with-each-prompt/765679](https://community.openai.com/t/correct-way-to-submit-the-db-schema-with-each-prompt/765679)  
23. Scraping-Screens-for-Popular-Investing-Themes-on-SCREENER ..., accessed January 17, 2026, [https://github.com/n-k-y/Scraping-Screens-for-Popular-Investing-Themes-on-SCREENER-using-Python/blob/main/project-web-scraping-with-python%20(1).ipynb](https://github.com/n-k-y/Scraping-Screens-for-Popular-Investing-Themes-on-SCREENER-using-Python/blob/main/project-web-scraping-with-python%20\(1\).ipynb)  
24. python \- can't scrape table element from <https://www.screener.in>, accessed January 17, 2026, [https://stackoverflow.com/questions/66719130/cant-scrape-table-element-from-https-www-screener-in](https://stackoverflow.com/questions/66719130/cant-scrape-table-element-from-https-www-screener-in)  
25. How to extract Screener.in screen results using Python \- YouTube, accessed January 17, 2026, [https://www.youtube.com/watch?v=YHg6W50l8Uc](https://www.youtube.com/watch?v=YHg6W50l8Uc)  
26. nsepython \- PyPI, accessed January 17, 2026, [https://pypi.org/project/nsepython/](https://pypi.org/project/nsepython/)  
27. Transcript of the Q3 2025-26 Earnings Conference Call held at 19 ..., accessed January 17, 2026, [https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q3/Management%20Commentary/Transcript%20of%20the%20Q3%202025-26%20Earnings%20Conference%20Call%20held%20on%20Jan%2012,%202026.pdf](https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/q3/Management%20Commentary/Transcript%20of%20the%20Q3%202025-26%20Earnings%20Conference%20Call%20held%20on%20Jan%2012,%202026.pdf)  
28. TCS to log a steady Q3 amid AI pivot, IT spending trend in focus, accessed January 17, 2026, [https://m.economictimes.com/markets/stocks/earnings/tcs-to-log-a-steady-q3-amid-ai-pivot-it-spending-trend-in-focus/articleshow/126473269.cms](https://m.economictimes.com/markets/stocks/earnings/tcs-to-log-a-steady-q3-amid-ai-pivot-it-spending-trend-in-focus/articleshow/126473269.cms)  
29. TCS Q3 FY26 presentation: Stable margins amid modest growth, AI ..., accessed January 17, 2026, [https://uk.investing.com/news/company-news/tcs-q3-fy26-presentation-stable-margins-amid-modest-growth-ai-focus-continues-93CH-4449220](https://uk.investing.com/news/company-news/tcs-q3-fy26-presentation-stable-margins-amid-modest-growth-ai-focus-continues-93CH-4449220)  
30. Strong TCV in a seasonally challenging Q3 positions TCS for Long ..., accessed January 17, 2026, [https://www.tcs.com/who-we-are/newsroom/press-release/tcs-financial-results-q3-fy-2025](https://www.tcs.com/who-we-are/newsroom/press-release/tcs-financial-results-q3-fy-2025)  
31. Mastering Tools and Tool Calling Agents in LangChain \- Medium, accessed January 17, 2026, [https://medium.com/@mariaaawaheed/mastering-tools-and-tool-calling-agents-in-langchain-a-comprehensive-guide-18a566f2aac5](https://medium.com/@mariaaawaheed/mastering-tools-and-tool-calling-agents-in-langchain-a-comprehensive-guide-18a566f2aac5)  
32. Tool Calling with LangChain, accessed January 17, 2026, [https://blog.langchain.com/tool-calling-with-langchain/](https://blog.langchain.com/tool-calling-with-langchain/)  
33. Build a financial AI search workflow using LangGraph.js ... \- Elastic, accessed January 17, 2026, [https://www.elastic.co/search-labs/blog/ai-agent-workflow-finance-langgraph-elasticsearch](https://www.elastic.co/search-labs/blog/ai-agent-workflow-finance-langgraph-elasticsearch)  
34. ChatGoogleGenerativeAI \- Docs by LangChain, accessed January 17, 2026, [https://docs.langchain.com/oss/python/integrations/chat/google\_generative\_ai](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)  
35. LangChain Google Gemini Integration: Complete Setup Guide \+ ..., accessed January 17, 2026, [https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-google-gemini-integration-complete-setup-guide-code-examples-2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langchain-setup-tools-agents-memory/langchain-google-gemini-integration-complete-setup-guide-code-examples-2025)
