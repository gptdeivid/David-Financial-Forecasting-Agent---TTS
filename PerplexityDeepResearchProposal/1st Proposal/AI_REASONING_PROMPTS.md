# TCS Forecast Agent: Complete AI Reasoning & Prompt Engineering Guide

## Executive Summary

This document details the **AI-first reasoning architecture** for the TCS Business Outlook Forecast Agent, including:

1. **Reasoning loops** (ReAct pattern with multi-tool orchestration)
2. **Master prompts** for each tool (extraction, RAG, synthesis)
3. **Guardrails** to prevent hallucination and ensure grounding
4. **Evaluation metrics** to validate forecast quality
5. **Error handling** strategies for edge cases

---

## Part 1: Core Reasoning Pattern (ReAct Agent)

### The Reason → Act → Observe Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│    "Generate Q4 FY25 business outlook for TCS"                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ AGENT THINKS (Reason) │
                    │                       │
                    │ "I need to extract:   │
                    │ - Latest revenue data │
                    │ - Profitability trend │
                    │ - Management sentiment│
                    │ - Client spending    │
                    │ - Macro headwinds"   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────────────────────┐
                    │ AGENT ACTS (Tool Invocation)          │
                    │                                       │
                    │ Tool 1: Extract Financial Data        │
                    │  - Input: Q3 FY25 Report PDF          │
                    │  - LLM: Extract metrics               │
                    │  - Output: Metrics JSON               │
                    │                                       │
                    │ Tool 2: Analyze Earnings Calls        │
                    │  - Input: "Management outlook?"       │
                    │  - Vector Search: Find relevant clips │
                    │  - LLM: Analyze sentiment             │
                    │  - Output: Insights JSON              │
                    │                                       │
                    │ Tool 3: Market Data (Optional)        │
                    │  - Input: TCS ticker                  │
                    │  - API: Fetch current price           │
                    │  - Output: Price, PE, yields          │
                    │                                       │
                    └───────────┬──────────────────────────┘
                                │
                    ┌───────────▼──────────────────────────┐
                    │ AGENT OBSERVES (Tool Output)         │
                    │                                      │
                    │ Financial Metrics:                   │
                    │ - Revenue: 62,613 Cr (Q3 FY25)       │
                    │ - Margin: 21.0%                      │
                    │ - YoY Growth: +5.4%                  │
                    │ - EPS: Rs 17.50                      │
                    │                                      │
                    │ Management Sentiment:                │
                    │ - Tone: Cautiously optimistic        │
                    │ - Key themes: AI growth, caution     │
                    │ - Risks: Client spending pause       │
                    │ - Opportunities: Digital services    │
                    │                                      │
                    │ Market Context:                      │
                    │ - Stock: $35.40, PE: 21              │
                    │ - Dividend: 2.8%                     │
                    │                                      │
                    └───────────┬──────────────────────────┘
                                │
                    ┌───────────▼───────────────────────┐
                    │ AGENT SYNTHESIZES (Reason Again)  │
                    │                                   │
                    │ "Revenue +5.4% YoY with stable    │
                    │  margins = strong operational     │
                    │  performance. But management     │
                    │  cautious on next quarter due    │
                    │  to client spending pause.       │
                    │  Opportunity in AI services.     │
                    │                                   │
                    │ Forecast: Expect 3-5% growth     │
                    │ Q4 with stable margins, offset    │
                    │ by macro caution."                │
                    │                                   │
                    └───────────┬───────────────────────┘
                                │
                    ┌───────────▼──────────────────────────────┐
                    │ OUTPUT: Structured Forecast JSON        │
                    │ {                                       │
                    │   "forecast_summary": "...",           │
                    │   "key_financial_trends": [...],       │
                    │   "management_outlook": "...",         │
                    │   "risks_and_opportunities": [...],    │
                    │   "source_documents": [...],           │
                    │   "errors": []                         │
                    │ }                                       │
                    └───────────────────────────────────────┘
```

### Key Principles of This Loop

1. **Reason First**: Agent decides what data is needed BEFORE calling tools
2. **Multi-Tool**: Tools called in sequence (financial → qualitative → market)
3. **Observe & Integrate**: Each tool output informs next decision
4. **Synthesize**: Final reasoning step combines all signals
5. **Iterate**: If conflict detected, loop back to investigate

---

## Part 2: Master Prompts for Each Tool

### Tool 1: FinancialDataExtractorTool Master Prompt

**Purpose**: Systematically extract quantitative metrics from financial reports

**Temperature**: 0.0 (deterministic - we want exact data, no variation)

```
SYSTEM PROMPT:
═══════════════════════════════════════════════════════════════════

You are an expert financial analyst extracting quantitative metrics 
from corporate financial statements. Your role is PRECISION and VERIFIABILITY.

You are analyzing: TCS (Tata Consultancy Services) quarterly financial reports.

CRITICAL RULE: ONLY extract metrics that are EXPLICITLY stated in the document.
Do NOT estimate, infer, or calculate. Extract AS-IS.

REQUIRED EXTRACTION TARGETS:
─────────────────────────────

1. REVENUE METRICS
   • Total Revenue (consolidated)
   • Revenue by segment (IT Services, Consulting, etc.)
   • YoY Growth %
   • QoQ Growth %
   • Geographic breakup (if available)

2. PROFITABILITY METRICS
   • Net Profit / Net Income
   • Operating Profit
   • Operating Margin %
   • Net Profit Margin %
   • EBITDA (if disclosed)
   • Tax Rate

3. PER-SHARE METRICS
   • Earnings Per Share (EPS)
   • Book Value Per Share
   • Dividend Per Share

4. CASH FLOW & BALANCE SHEET
   • Operating Cash Flow
   • Free Cash Flow
   • Total Assets
   • Total Liabilities
   • Shareholder Equity

5. EFFICIENCY & RATIOS
   • Return on Equity (ROE)
   • Asset Turnover
   • Employee Productivity
   • Attrition Rate

EXTRACTION PROTOCOL:
─────────────────────

For EACH metric extracted:

a) STATE THE VALUE
   Example: "Revenue: Rs 62,613 Crore"

b) SPECIFY THE PERIOD
   Example: "Q3 FY2025" or "9 Months FY2025"

c) PROVIDE DIRECT QUOTE
   Example: "Source quote from page 3: 'Revenue of Rs 62,613 crore 
            in the first nine months of fiscal 2025, up 5.4% year-on-year'"

d) CITE LOCATION
   Example: "From Consolidated Statement of Profit & Loss, page 5"

e) ASSIGN CONFIDENCE
   - 1.0: Explicitly bolded/in prominent table
   - 0.8: Clearly stated in body text, unambiguous
   - 0.6: Stated but requires minor interpretation
   - 0.4: Partially inferred from context
   - 0.0: Not found or requires significant inference (EXCLUDE)

f) VERIFY LOGIC
   Check internal consistency:
   • Revenue > 0
   • Margin = (Profit / Revenue) ± 2% tolerance
   • Growth rate reasonable (-50% to +50%)

QUALITY GATES:
──────────────

❌ EXCLUDE metrics with:
  - Confidence < 0.6
  - No direct source quote
  - Conflicting values across document sections
  - Logical inconsistencies

✓ INCLUDE metrics with:
  - Confidence ≥ 0.6
  - Direct, verifiable quotes
  - Consistent across document
  - Logical coherence

OUTPUT FORMAT:
───────────────

Return a JSON array of extracted metrics, strictly adhering to this schema:

[
  {
    "metric_name": string,        // e.g., "Total Revenue"
    "value": float,               // numeric value only
    "unit": string,               // e.g., "INR Crore", "%"
    "period": string,             // e.g., "Q3 FY25", "9M FY25"
    "yoy_change": float | null,   // percentage change
    "source_quote": string,       // Exact quote from document
    "page_reference": string,     // Page number where found
    "document_section": string,   // e.g., "Consolidated P&L"
    "confidence": float,          // 0.0 to 1.0
    "notes": string | null        // Any ambiguity or context
  }
]

EXAMPLES:
──────────

✓ CORRECT:
{
  "metric_name": "Total Revenue",
  "value": 62613,
  "unit": "INR Crore",
  "period": "Q3 FY25",
  "yoy_change": 5.4,
  "source_quote": "Revenue of Rs 62,613 crore in the first nine 
                   months of fiscal 2025, up 5.4% year-on-year",
  "page_reference": "3",
  "document_section": "MD&A",
  "confidence": 1.0,
  "notes": null
}

✗ WRONG (should be excluded):
{
  "metric_name": "Estimated Q4 Revenue",    // ← ESTIMATED (not stated)
  "value": 65000,                            // ← Inferred
  "confidence": 0.3,                         // ← Below threshold
  "notes": "Calculated by dividing 9M revenue by 3"  // ← Inference
}

IMPORTANT REMINDERS:
─────────────────────

• Precision over completeness: 5 accurate metrics > 10 hallucinated ones
• Every number must be traceable to source text
• When in doubt, exclude the metric
• Confidence scoring is crucial for downstream filtering
• Cite verbatim quotes to enable verification
```

### Tool 2: QualitativeAnalysisTool Master Prompt (RAG-Based)

**Purpose**: Analyze earnings call transcripts for management outlook and sentiment

**Temperature**: 0.2 (low randomness, consistent interpretation)

```
SYSTEM PROMPT:
═══════════════════════════════════════════════════════════════════

You are analyzing TCS earnings call transcripts to identify forward-looking 
insights, management sentiment, and business drivers.

CONTEXT:
You have been provided with relevant excerpts from recent TCS earnings calls.
ONLY use information from these provided segments. Do NOT rely on external knowledge.

ANALYSIS OBJECTIVES:
──────────────────────

1. MANAGEMENT OUTLOOK
   • Is management guiding up, down, or flat for next quarter?
   • What's their tone? (Optimistic, cautious, uncertain)
   • Any explicit forward guidance statements?
   • Confidence in guidance? (High/medium/low)

2. KEY THEMES & PRIORITIES
   • What topics dominate the discussion?
   • How often are themes mentioned? (frequency = priority)
   • Segment-specific drivers (IT Services, Consulting, etc.)
   • Geographic focus areas

3. CLIENT DYNAMICS
   • What's happening with client spending?
   • Are clients investing or pausing?
   • Which verticals/geographies most/least healthy?
   • Hiring pace and attrition trends?

4. TECHNOLOGY & OPPORTUNITIES
   • What technology opportunities excite management?
   • AI/Cloud/Digital progress?
   • Investment priorities?

5. RISKS & HEADWINDS
   • Explicitly mentioned risks?
   • Macro concerns (geopolitical, economic)?
   • Competitive threats?
   • Internal challenges (talent, execution)?

6. SENTIMENT ANALYSIS
   • Overall sentiment: Positive | Neutral | Negative | Mixed
   • Sentiment by theme (e.g., "positive on AI, cautious on spending")
   • Tone consistency (has it shifted quarter-over-quarter)?

EXTRACTION PROTOCOL:
──────────────────────

For each theme identified:

a) STATE THE THEME
   Example: "Client Spending Caution"

b) ASSIGN SENTIMENT
   Positive, Neutral, Negative, or Mixed

c) PROVIDE DIRECT QUOTES (minimum 2)
   Example: 
   [Quote 1] "We're seeing our clients take a more cautious stance"
   [Quote 2] "Discretionary spending has come to a pause in Q3"

d) ESTIMATE PRIORITY (frequency of mention)
   Low (1x) | Medium (2-3x) | High (4+ times)

e) ASSESS FORECAST RELEVANCE
   How does this theme impact next quarter outlook?
   Impact: High | Medium | Low

f) CROSS-CALL CONSISTENCY
   Is this theme consistent across multiple earnings calls?
   Consistent | Emerging | Changing | One-off mention

OUTPUT FORMAT:
───────────────

Return a JSON object strictly adhering to this schema:

{
  "analysis_date": string,                    // ISO date of analysis
  "transcript_dates": [string],               // Dates of calls analyzed
  "themes": [
    {
      "theme": string,                        // Theme identifier
      "sentiment": string,                    // Positive/Neutral/Negative/Mixed
      "direct_quotes": [string],              // Array of verbatim quotes
      "mention_frequency": string,            // Low/Medium/High
      "forecast_relevance": string,           // Impact on Q4 forecast
      "cross_call_consistency": string,       // How consistent across calls
      "supporting_evidence": string,          // Brief explanation
      "confidence": float                     // 0.0 to 1.0
    }
  ],
  "overall_sentiment": string,                // Positive/Neutral/Negative/Mixed
  "management_guidance": string,              // Any explicit forward guidance
  "top_3_themes": [string],                   // Most important themes
  "identified_risks": [string],               // Explicit risks mentioned
  "identified_opportunities": [string],       // Opportunities highlighted
  "quarter_over_quarter_sentiment_change": string  // Improving/Stable/Declining
}

EXAMPLES:
──────────

✓ CORRECT THEME EXTRACTION:
{
  "theme": "Client Spending Pause",
  "sentiment": "Negative",
  "direct_quotes": [
    "We're seeing our clients take a more cautious stance on discretionary 
     spending, particularly from BFSI clients",
    "The macro environment has made clients more deliberate with their IT 
     spending decisions"
  ],
  "mention_frequency": "High",
  "forecast_relevance": "HIGH - Directly impacts Q4 growth expectations",
  "cross_call_consistency": "Consistent - mentioned in both Q2 and Q3 calls",
  "supporting_evidence": "This theme appears 5 times across calls, with specific 
                          mentions from CEO and CFO. Not a one-off comment.",
  "confidence": 0.95
}

✗ WRONG (should be questioned):
{
  "theme": "Uncertain Market Outlook",
  "direct_quotes": [
    "Management didn't explicitly comment on Q4 outlook"  // ← NOT A QUOTE
  ],
  "confidence": 0.2                                       // ← Too low to include
}

QUALITY GATES:
──────────────

❌ EXCLUDE themes with:
  - Confidence < 0.7
  - No direct quotes from transcript
  - Mentioned only once (one-off comments)
  - Speculative or inferential
  - Not relevant to business forecast

✓ INCLUDE themes with:
  - Confidence ≥ 0.7
  - Direct verbatim quotes
  - Mentioned 2+ times (indicates priority)
  - Grounded in explicit statements
  - Clear business relevance

CRITICAL REMINDERS:
────────────────────

1. ALL quotes must match transcript text EXACTLY
2. Use Ctrl+F search to verify quote accuracy
3. Speaker attribution helpful but not required
4. Focus on FORWARD-LOOKING statements (guidance, outlook)
5. Don't infer; use only explicit information
6. Confidence reflects clarity of signal, not certainty of event
```

### Tool 3: Agent Synthesis Prompt (Forecast Generation)

**Purpose**: Combine quantitative and qualitative signals into coherent forecast

**Temperature**: 0.4 (balanced - need creativity but grounded in data)

```
SYSTEM PROMPT:
═══════════════════════════════════════════════════════════════════

You are synthesizing a quarterly business outlook forecast for TCS based on:
1. Extracted financial metrics (quantitative)
2. Management sentiment from earnings calls (qualitative)
3. Current market context (optional)

Your task is to produce a REASONED, DATA-DRIVEN forecast that:
• Integrates both quantitative and qualitative signals
• Identifies and reconciles any conflicts in signals
• Provides clear rationale for each forecast element
• Cites sources for all claims
• Acknowledges uncertainty and limitations

INPUT DATA:
───────────────

Financial Metrics (from FinancialDataExtractorTool):
{
  "revenue_q3": 62613,
  "revenue_growth_yoy": 5.4,
  "operating_margin": 21.0,
  "eps": 17.50,
  ...
}

Management Insights (from QualitativeAnalysisTool):
{
  "themes": [
    {
      "theme": "Client Spending Caution",
      "sentiment": "Negative",
      "quotes": ["..."],
      "forecast_relevance": "HIGH"
    },
    {
      "theme": "AI Services Momentum",
      "sentiment": "Positive",
      "quotes": ["..."],
      "forecast_relevance": "HIGH"
    }
  ],
  "overall_sentiment": "Mixed"
}

SYNTHESIS FRAMEWORK:
──────────────────────

Step 1: ASSESS QUANTITATIVE TREND
  • Q3 revenue: 62,613 Cr (+5.4% YoY)
  • Q3 margin: 21.0% (stable vs prior quarters)
  • Trend analysis: Growing but decelerating?
  • Trajectory: What does historical trend suggest for Q4?

Step 2: ASSESS QUALITATIVE SIGNALS
  • Key theme: Client spending caution (HIGH relevance, NEGATIVE sentiment)
  • Counter-theme: AI services demand (HIGH relevance, POSITIVE sentiment)
  • Net signal: Mixed, but caution dominates near-term
  • Forecast implication: Growth may moderate Q4

Step 3: CROSS-VALIDATE SIGNALS
  ┌─────────────────────────────────────────────────────┐
  │ Potential Conflict Analysis:                        │
  │                                                     │
  │ Quant says: Growing 5.4% YoY (positive)            │
  │ Qual says: Clients cautious (negative)             │
  │                                                     │
  │ Resolution: "Recent quarter strong, but forward    │
  │ guidance cautious. Suggests near-term momentum     │
  │ decelerating as client spending pause takes effect."│
  │                                                     │
  │ Evidence:                                           │
  │ • Mgmt explicitly stated client caution 3+ times   │
  │ • Guidance "seeing pause in discretionary"        │
  │ • Implies Q4 slower than Q3                       │
  └─────────────────────────────────────────────────────┘

Step 4: GENERATE FORECAST ELEMENTS
  
  a) FORECAST SUMMARY
  "TCS expects revenue growth to moderate to 3-5% in Q4 FY25 from 5.4% 
   in Q3, driven by client spending caution. However, margin resilience 
   and AI services momentum provide upside offset."

  b) KEY FINANCIAL TRENDS
  [Bullet points based on extracted metrics + growth trajectory]
  - Q3 Revenue: Rs 62,613 Cr, +5.4% YoY
  - Q3 Margin: 21.0%, stable despite growth moderation
  - Trend: Growth decelerating into Q4
  - Opportunity: AI services growing double-digit (per mgmt)

  c) MANAGEMENT OUTLOOK
  "Management acknowledged near-term client spending caution but expressed 
   confidence in long-term AI opportunity. Tone was cautiously optimistic 
   about medium-term prospects despite Q4 headwinds."
  
  [Include 2-3 direct quotes to substantiate]

  d) RISKS & OPPORTUNITIES
  Risks:
  - Client spending pause may extend beyond Q4 (mgmt cautious tone)
  - Macro headwinds (geopolitical, economic slowdown)
  - BFSI specifically mentioned as cautious (largest vertical)
  
  Opportunities:
  - AI services demand strong (mgmt mentioned 4+ times)
  - Digital transformation acceleration
  - Cloud migration continuing

Step 5: DOCUMENT SOURCE CITATIONS
  For every claim, cite:
  • Source document (Q3 Results, Earnings Call)
  • Page/section reference
  • Verbatim quote if extractive claim
  • LLM-synthesized if interpretive

OUTPUT FORMAT:
──────────────

{
  "forecast_summary": string,           // 2-3 sentence outlook
  "key_financial_trends": [string],     // 3-5 data-driven trends
  "management_outlook": string,         // 2-3 sentence sentiment summary
                                        // with supporting quotes
  "risks_and_opportunities": {
    "risks": [string],                  // 2-3 explicit risks
    "opportunities": [string]           // 2-3 explicit opportunities
  },
  "source_documents": [string],         // URLs/paths of documents used
  "reasoning": {
    "quantitative_signal": string,      // What numbers suggest
    "qualitative_signal": string,       // What mgmt suggests
    "conflict_analysis": string | null, // If signals conflict, how resolved
    "confidence_level": string,         // High/Medium/Low
    "key_assumptions": [string]         // Assumptions underlying forecast
  },
  "errors": []                          // Any issues encountered
}

EXAMPLE OUTPUT:
────────────────

{
  "forecast_summary": "TCS expects moderate revenue growth of 3-5% in Q4 
                       FY25, decelerating from Q3's 5.4% due to client 
                       spending caution identified by management. Margins 
                       expected to remain stable at ~21% despite macro 
                       headwinds.",
  
  "key_financial_trends": [
    "Q3 FY25 Revenue: Rs 62,613 Cr (+5.4% YoY); trend decelerating",
    "Operating Margin: 21% stable; limited margin expansion expected Q4",
    "Client Spending: Explicitly cautious per mgmt (mentioned 5x in call)",
    "AI Services: Strong demand; positioned as key growth driver",
    "EPS Trend: Rs 17.50 (Q3); modest growth expected Q4 (5-7%)"
  ],
  
  "management_outlook": "Management tone 'cautiously optimistic.' 
                        Acknowledged 'pause in discretionary spending' 
                        but expressed confidence in AI opportunity and 
                        medium-term trajectory. Quote: 'We're seeing 
                        clients take deliberate approach but not reducing 
                        investments.'",
  
  "risks_and_opportunities": {
    "risks": [
      "Client spending pause may persist beyond Q4 if macro worsens",
      "BFSI caution (explicitly called out, TCS's largest vertical)",
      "Attrition remains elevated; talent retention risk"
    ],
    "opportunities": [
      "AI services momentum (mgmt confidence, explicit growth guidance)",
      "Digital transformation accelerating in key verticals",
      "Cloud adoption continuing as structural trend"
    ]
  },
  
  "source_documents": [
    "TCS Q3 FY25 Results - Investor Release (Oct 2024)",
    "TCS Q3 FY25 Earnings Call Transcript (Oct 2024)"
  ],
  
  "reasoning": {
    "quantitative_signal": "5.4% YoY growth with stable margins suggests 
                            operational strength, but 9M data (combined 
                            Q1-Q3) shows deceleration from 7% in earlier 
                            quarters → Q4 may slow further",
    
    "qualitative_signal": "Management highlighted client spending caution 
                          3+ times. This is explicit guidance on near-term 
                          pressure. Offset by confidence in AI (4+ mentions) 
                          and medium-term growth.",
    
    "conflict_analysis": "Quant shows growth momentum; qual suggests 
                         deceleration. Resolution: 'Recent quarter benefited 
                         from Q2/Q3 large deals; Q4 likely slower due to 
                         client pause starting to impact. Growth moderates 
                         3-5%.'",
    
    "confidence_level": "Medium-High on direction (growth moderates), 
                        Medium on magnitude (exact 3-5% range uncertain)",
    
    "key_assumptions": [
      "Client spending pause affects Q4 but doesn't reverse",
      "Margins held despite growth moderation",
      "AI services offset general slowdown by 1-2%",
      "Macro doesn't significantly worsen"
    ]
  },
  
  "errors": []
}

GUARDRAILS:
────────────

1. CONFLICT DETECTION:
   IF quantitative and qualitative signals DIVERGE:
   → Investigate in "conflict_analysis"
   → Provide reasoned reconciliation
   → Flag confidence as "Medium" if conflict unresolved

2. HALLUCINATION PREVENTION:
   IF generating forecast element WITHOUT data support:
   → Mark as "assumption" not "observation"
   → Flag confidence accordingly
   → Cite source if available

3. CONFIDENCE CALIBRATION:
   • High: Explicit mgmt guidance + consistent data
   • Medium: Data supports trend but exceptions exist
   • Low: Signals mixed, significant uncertainty

4. SOURCE TRACEABILITY:
   Every forecast element should map to:
   • Extracted metric (with confidence score), OR
   • Management quote (with date/speaker), OR
   • Reasonable inference (flagged as such)
```

---

## Part 3: Guardrails & Safety Mechanisms

### Guardrail 1: Source Verification

```python
# Pseudo-code for verification logic

def verify_extraction(claim: str, source_document: str) -> VerificationResult:
    """
    Every extracted metric must have a verifiable source.
    
    Process:
    1. Extract quoted text from claim
    2. Search for exact match in source document
    3. If found: confidence++ (verified)
    4. If not found: mark as hallucination, exclude
    """
    
    quote = claim["source_quote"]
    
    if quote in source_document:
        return VerificationResult(
            verified=True,
            confidence=claim["confidence"] * 1.1  # Boost confidence
        )
    else:
        logger.error(f"HALLUCINATION DETECTED: Quote not in source")
        logger.error(f"Claimed quote: {quote}")
        
        return VerificationResult(
            verified=False,
            confidence=0.0,
            action="EXCLUDE"
        )
```

### Guardrail 2: Confidence Scoring

```
Extraction Confidence Scale:

1.0 │ ███████████████ EXPLICITLY BOLDED/HIGHLIGHTED
    │                Unambiguous statement in prominent location
    │
0.8 │ █████████████░░ CLEARLY STATED
    │                Stated in body text, no interpretation needed
    │
0.6 │ ███████░░░░░░░ REQUIRES MINOR INTERPRETATION
    │                Context clues needed, but reasonable
    │
0.4 │ ████░░░░░░░░░░ PARTIALLY INFERRED
    │                Requires significant context interpretation
    │
0.0 │ ░░░░░░░░░░░░░░ NOT FOUND / HALLUCINATED
    │                EXCLUDE from forecast
    └─────────────────────────────────────────

Decision Rule:
- Include in forecast: Confidence ≥ 0.6
- Flag for review: Confidence 0.4-0.6
- Exclude: Confidence < 0.4
```

### Guardrail 3: Conflict Detection & Resolution

```python
class ConflictDetector:
    """Detect and reconcile conflicting signals."""
    
    def detect_conflicts(self, quant_signal, qual_signal) -> list:
        """
        Returns conflicts with severity levels.
        """
        
        conflicts = []
        
        # Pattern 1: Revenue growth positive but sentiment negative
        if quant_signal["growth"] > 0 and qual_signal["sentiment"] == "negative":
            conflicts.append({
                "type": "TONE_MISMATCH",
                "severity": "HIGH",
                "description": "Growing revenue but mgmt cautious",
                "possible_explanations": [
                    "One-time items inflating recent growth",
                    "Forward guidance reflecting near-term caution",
                    "Margin pressure hiding volume growth",
                    "Geographic mix shift"
                ]
            })
        
        # Pattern 2: Margin stability with declining growth
        if quant_signal["margin_trend"] == "stable" and quant_signal["growth"] < quant_signal["prior_growth"]:
            conflicts.append({
                "type": "MARGIN_RESILIENCE",
                "severity": "MEDIUM",
                "description": "Margins stable despite growth deceleration",
                "possible_explanations": [
                    "Mix improvement offsetting volume decline",
                    "Cost control offsetting lower volume",
                    "Pricing power maintained"
                ]
            })
        
        return conflicts
    
    def resolve_conflicts(self, conflicts: list) -> ResolutionStrategy:
        """
        Generate resolution strategy for each conflict.
        """
        
        for conflict in conflicts:
            if conflict["severity"] == "HIGH":
                # Investigate further
                investigation = self.investigate(conflict)
                
                # Choose most likely explanation
                most_likely = max(
                    investigation["explanations"],
                    key=lambda x: x["probability"]
                )
                
                return ResolutionStrategy(
                    conflict=conflict["type"],
                    resolution=most_likely,
                    confidence=most_likely["probability"],
                    action="SYNTHESIZE_WITH_CAVEAT"
                )
```

### Guardrail 4: Hallucination Detection

```python
class HallucinationGuard:
    """Prevent LLM from inventing data."""
    
    def screen_extraction(self, extraction: dict, source_text: str) -> dict:
        """
        Validate all extracted claims before including in forecast.
        """
        
        hallucinations = []
        
        for claim in extraction.get("metrics", []):
            # Test 1: Source quote exists
            if claim["source_quote"] not in source_text:
                hallucinations.append({
                    "claim": claim["metric_name"],
                    "issue": "QUOTE_NOT_IN_SOURCE",
                    "action": "EXCLUDE"
                })
                continue
            
            # Test 2: Value plausible
            if not self._is_plausible(claim):
                hallucinations.append({
                    "claim": claim["metric_name"],
                    "issue": "VALUE_IMPLAUSIBLE",
                    "example": f"{claim['value']} {claim['unit']}",
                    "action": "FLAG_FOR_REVIEW"
                })
                continue
            
            # Test 3: Consistency check
            if not self._is_consistent(claim, extraction):
                hallucinations.append({
                    "claim": claim["metric_name"],
                    "issue": "INCONSISTENT_WITH_CONTEXT",
                    "example": f"Claims margin=25% but profit/revenue=18%",
                    "action": "FLAG_FOR_REVIEW"
                })
        
        return {
            "clean_extraction": [c for c in extraction["metrics"] 
                               if c["metric_name"] not in [h["claim"] for h in hallucinations]],
            "hallucinations": hallucinations,
            "quality_score": (len(extraction["metrics"]) - len(hallucinations)) / len(extraction["metrics"])
        }
    
    def _is_plausible(self, claim):
        """Check if extracted value is realistic."""
        
        # Revenue > 0
        if claim["metric_name"] == "Revenue" and claim["value"] <= 0:
            return False
        
        # Growth between -50% and +50%
        if "growth" in claim["metric_name"].lower():
            if not -0.5 <= claim["value"] <= 0.5:
                return False
        
        # Margin between 0% and 100%
        if "margin" in claim["metric_name"].lower():
            if not 0 <= claim["value"] <= 1:
                return False
        
        return True
```

---

## Part 4: Evaluation & Quality Metrics

### Metric 1: Source Fidelity

```python
def evaluate_source_fidelity(forecast: dict, documents: dict) -> float:
    """
    Score: 0-1
    Measures: % of claims verifiable in source documents
    
    Calculation:
    - Count total claims in forecast
    - For each claim, verify source quote exists
    - Score = verified_claims / total_claims
    """
    
    claims = extract_claims(forecast)
    verified = 0
    
    for claim in claims:
        for doc in documents.values():
            if claim["source_quote"] in doc:
                verified += 1
                break
    
    score = verified / len(claims) if claims else 1.0
    
    return {
        "metric": "source_fidelity",
        "score": score,
        "status": "GOOD" if score >= 0.9 else "ACCEPTABLE" if score >= 0.75 else "REVIEW_NEEDED"
    }
```

### Metric 2: Confidence Calibration

```python
def evaluate_confidence_calibration(forecast: dict) -> dict:
    """
    Score: 0-1
    Measures: Are confidence scores realistic?
    
    Tests:
    - Claims with confidence < 0.6 shouldn't appear in forecast
    - Claims with high confidence should be heavily quoted
    - Forecast confidence should align with average component confidence
    """
    
    issues = []
    
    # Check for low-confidence claims
    for claim in forecast["claims"]:
        if claim["confidence"] < 0.6:
            issues.append(f"Low-confidence claim included: {claim['metric']}")
    
    # Check for unsupported high-confidence claims
    for claim in forecast["claims"]:
        if claim["confidence"] >= 0.9 and len(claim.get("source_quotes", [])) < 2:
            issues.append(f"High-confidence claim with insufficient support: {claim['metric']}")
    
    # Overall calibration
    avg_confidence = np.mean([c["confidence"] for c in forecast["claims"]])
    forecast_confidence = forecast.get("confidence")
    
    if abs(avg_confidence - forecast_confidence) > 0.2:
        issues.append(f"Forecast confidence miscalibrated: "
                     f"components={avg_confidence:.2f}, forecast={forecast_confidence:.2f}")
    
    return {
        "metric": "confidence_calibration",
        "issues": issues,
        "status": "GOOD" if len(issues) == 0 else "NEEDS_REVIEW"
    }
```

### Metric 3: Forecast Accuracy (Post-Quarter)

```python
def evaluate_forecast_accuracy(forecast: dict, actuals: dict) -> dict:
    """
    Score: 0-1 (after actual results available)
    Measures: How close was forecast to actual results?
    
    Calculates Mean Absolute Percentage Error (MAPE)
    """
    
    errors = {}
    forecast_metrics = ["revenue_growth", "margin", "eps_growth"]
    
    for metric in forecast_metrics:
        if metric in forecast and metric in actuals:
            forecast_val = forecast[metric]
            actual_val = actuals[metric]
            
            # Calculate percentage error
            error = abs((forecast_val - actual_val) / actual_val) * 100
            errors[metric] = error
    
    mape = np.mean(list(errors.values())) if errors else None
    
    # Determine accuracy rating
    if mape is None:
        accuracy_rating = "INSUFFICIENT_DATA"
    elif mape < 10:
        accuracy_rating = "EXCELLENT"
    elif mape < 20:
        accuracy_rating = "GOOD"
    elif mape < 30:
        accuracy_rating = "ACCEPTABLE"
    else:
        accuracy_rating = "NEEDS_IMPROVEMENT"
    
    return {
        "metric": "forecast_accuracy",
        "mape": mape,
        "errors_by_component": errors,
        "accuracy_rating": accuracy_rating,
        "direction_accuracy": evaluate_direction(forecast, actuals)
    }
```

---

## Part 5: Error Handling & Recovery

### Pattern 1: Tool Execution Failure with Retry

```python
async def execute_tool_with_retry(
    tool_func,
    *args,
    max_retries=3,
    base_delay=1.0
):
    """
    Execute tool with exponential backoff retry logic.
    """
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Executing {tool_func.__name__} (attempt {attempt+1}/{max_retries})")
            result = await tool_func(*args)
            logger.info(f"Success on attempt {attempt+1}")
            return result
        
        except (asyncio.TimeoutError, ConnectionError, RateLimitError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Max retries exceeded for {tool_func.__name__}")
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Tool failed: {str(e)}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
        
        except Exception as e:
            logger.error(f"Unexpected error in {tool_func.__name__}: {str(e)}", exc_info=True)
            raise
```

### Pattern 2: Graceful Degradation on Partial Failure

```python
async def generate_forecast_with_degradation(company: str) -> dict:
    """
    Attempt full forecast; gracefully degrade if tools fail.
    """
    
    tools_status = {
        "financial": None,
        "qualitative": None,
        "market_data": None
    }
    
    # Tool 1: Financial Data
    try:
        financial_data = await financial_extractor()
        tools_status["financial"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Financial extraction failed: {e}")
        financial_data = None
        tools_status["financial"] = "FAILED"
    
    # Tool 2: Qualitative Analysis
    try:
        qualitative_data = await qualitative_analyzer()
        tools_status["qualitative"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Qualitative analysis failed: {e}")
        qualitative_data = None
        tools_status["qualitative"] = "FAILED"
    
    # Tool 3: Market Data (optional)
    try:
        market_data = await market_data_fetcher()
        tools_status["market_data"] = "SUCCESS"
    except Exception as e:
        logger.warning(f"Market data fetch failed: {e}")
        market_data = None
        tools_status["market_data"] = "FAILED"
    
    # Determine degradation level
    if tools_status["financial"] == "SUCCESS" and tools_status["qualitative"] == "SUCCESS":
        # Full forecast possible
        forecast = synthesize_full_forecast(financial_data, qualitative_data, market_data)
        forecast_level = "FULL"
    
    elif tools_status["financial"] == "SUCCESS":
        # Quantitative-only forecast
        forecast = synthesize_quantitative_forecast(financial_data)
        forecast["warning"] = "Qualitative analysis unavailable; forecast based on metrics only"
        forecast_level = "QUANTITATIVE_ONLY"
    
    elif tools_status["qualitative"] == "SUCCESS":
        # Qualitative-only forecast (rare)
        forecast = synthesize_qualitative_forecast(qualitative_data)
        forecast["warning"] = "Financial data unavailable; forecast based on management guidance only"
        forecast_level = "QUALITATIVE_ONLY"
    
    else:
        # All tools failed
        forecast = {
            "forecast_summary": "Unable to generate forecast; all data sources failed",
            "errors": ["Financial extraction failed", "Qualitative analysis failed"]
        }
        forecast_level = "UNAVAILABLE"
    
    return {
        "forecast": forecast,
        "degradation_level": forecast_level,
        "tools_status": tools_status
    }
```

---

## Summary: Complete Reasoning Pipeline

```
INPUT: "Generate Q4 FY25 forecast for TCS"
    │
    ├─→ [Agent Reason]
    │   "Need financial data + management sentiment"
    │
    ├─→ [Tool 1: Financial Extractor]
    │   Input: Q3 Results PDF
    │   LLM Prompt: (Master Prompt 1, temp=0.0)
    │   Output: Metrics JSON with confidence scores
    │   Guardrail: Source verification, hallucination check
    │
    ├─→ [Tool 2: Qualitative Analyzer]
    │   Input: "What's management outlook?"
    │   Vector Search: Retrieve earnings call segments
    │   LLM Prompt: (Master Prompt 2, temp=0.2)
    │   Output: Themes, sentiment, quotes
    │   Guardrail: Quote verification, consistency check
    │
    ├─→ [Tool 3: Market Data] (optional)
    │   Input: TCS ticker
    │   API: Fetch current price, ratios
    │   Output: Market context
    │
    ├─→ [Agent Synthesize]
    │   Input: Financial + Qualitative + Market data
    │   LLM Prompt: (Master Prompt 3, temp=0.4)
    │   Process:
    │     1. Cross-validate signals
    │     2. Detect conflicts
    │     3. Reconcile conflicts
    │     4. Generate narrative forecast
    │     5. Cite sources
    │   Guardrail: Conflict detection, confidence calibration
    │
    ├─→ [Structure Output]
    │   JSON schema validation
    │   Source traceability check
    │   Confidence scoring
    │
    ├─→ [Evaluate Quality]
    │   Metric 1: Source fidelity (% claims verifiable)
    │   Metric 2: Confidence calibration
    │   Metric 3: Forecast coherence
    │
    └─→ OUTPUT: Structured Forecast JSON
        {
          "forecast_summary": "...",
          "key_financial_trends": [...],
          "management_outlook": "...",
          "risks_and_opportunities": [...],
          "source_documents": [...],
          "errors": []
        }
        
        PLUS:
        - MySQL log entry
        - Quality metrics
        - Reasoning trace
```

---

## Key Takeaways

1. **Temperature Control**: Use 0.0 for extraction (deterministic), 0.2-0.4 for reasoning (balanced)

2. **Source Verification**: Every claim must be traceable; hallucinations excluded ruthlessly

3. **Confidence Scoring**: 0-1 scale; < 0.6 excluded; guides downstream filtering

4. **Conflict Resolution**: When quant ≠ qual, investigate and reconcile explicitly

5. **Graceful Degradation**: If one tool fails, continue with available data; don't fail entirely

6. **Complete Audit Trail**: Every decision logged with source, confidence, timestamp

7. **Quality Gates**: Three layers of evaluation (source fidelity, confidence calibration, forecast coherence)

This architecture ensures **transparent, auditable, grounded reasoning** suitable for financial forecasting where accuracy and reproducibility matter.
