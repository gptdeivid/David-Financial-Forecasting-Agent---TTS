# File: app/main.py
# Complete starter implementation for FastAPI TCS Forecast Agent

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import logging
import json
import time

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiomysql
from dotenv import load_dotenv
import os

from app.config import Settings
from app.services.agent import AgentService
from app.services.logger import AsyncDatabaseLogger
from app.models.forecast import ForecastRequest, ForecastResponse

# Load environment variables
load_dotenv()
settings = Settings()

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Lifespan Management (FastAPI 0.93+)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan:
    - On startup: Initialize services
    - On shutdown: Clean up resources
    """
    logger.info("🚀 Application starting up...")
    
    # Initialize MySQL connection pool
    app.state.mysql_pool = await aiomysql.create_pool(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        db=settings.mysql_database,
        minsize=5,
        maxsize=20,
        autocommit=True
    )
    logger.info("✓ MySQL pool created")
    
    # Initialize agent service
    app.state.agent_service = AgentService(settings)
    logger.info("✓ Agent service initialized")
    
    # Initialize logger service
    app.state.logger_service = AsyncDatabaseLogger(app.state.mysql_pool)
    logger.info("✓ Database logger initialized")
    
    logger.info("✓ All services ready")
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    app.state.mysql_pool.close()
    await app.state.mysql_pool.wait_closed()
    logger.info("✓ Cleanup complete")


# ─────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="TCS Business Outlook Forecast Agent",
    description="AI-powered forecasting service for Tata Consultancy Services",
    version="1.0.0",
    lifespan=lifespan
)


# ─────────────────────────────────────────────────────────────
# Dependency Injection Functions
# ─────────────────────────────────────────────────────────────

async def get_agent_service() -> AgentService:
    """Inject agent service."""
    return app.state.agent_service

async def get_logger_service() -> AsyncDatabaseLogger:
    """Inject database logger service."""
    return app.state.logger_service


# ─────────────────────────────────────────────────────────────
# Health Check Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        dict: Status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# ─────────────────────────────────────────────────────────────
# Main Forecast Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/forecast", response_model=dict, tags=["forecast"])
async def generate_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service),
    logger_service: AsyncDatabaseLogger = Depends(get_logger_service)
):
    """
    Generate quarterly business outlook forecast for TCS.
    
    This endpoint:
    1. Validates the request
    2. Invokes the multi-tool AI agent
    3. Structures the response according to API schema
    4. Logs the request/response asynchronously
    5. Returns structured forecast JSON
    
    Args:
        request: ForecastRequest with company and optional parameters
        background_tasks: FastAPI background task runner
        agent_service: Injected agent orchestrator
        logger_service: Injected async database logger
    
    Returns:
        ForecastResponse: Structured JSON with forecast, trends, insights
    
    Raises:
        HTTPException: 400 for invalid company, 500 for processing errors
    
    Example:
        POST /forecast
        {
          "company": "TCS",
          "include_market_data": true
        }
        
        Response (200):
        {
          "forecast_summary": "...",
          "key_financial_trends": [...],
          "management_outlook": "...",
          "risks_and_opportunities": [...],
          "source_documents": [...],
          "errors": []
        }
    """
    
    start_time = time.time()
    
    try:
        # ─────────────────────────────────────────────────────────────
        # Step 1: Validate Input
        # ─────────────────────────────────────────────────────────────
        
        if request.company.upper() != "TCS":
            logger.warning(f"Invalid company requested: {request.company}")
            raise HTTPException(
                status_code=400,
                detail="Currently only TCS is supported"
            )
        
        logger.info(
            f"Forecast request received for {request.company} | "
            f"market_data={request.include_market_data}"
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: Invoke Agent
        # ─────────────────────────────────────────────────────────────
        
        forecast_result = await agent_service.generate_forecast(
            company=request.company,
            include_market_data=request.include_market_data
        )
        
        logger.info("Agent forecast generation completed successfully")
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: Structure Response
        # ─────────────────────────────────────────────────────────────
        
        response = {
            "forecast_summary": forecast_result.get("summary", ""),
            "key_financial_trends": forecast_result.get("trends", []),
            "management_outlook": forecast_result.get("outlook", ""),
            "risks_and_opportunities": forecast_result.get("risks", []),
            "source_documents": forecast_result.get("sources", []),
            "errors": forecast_result.get("errors", [])
        }
        
        # Include market data if available
        if request.include_market_data and "market_data" in forecast_result:
            response["market_data"] = forecast_result["market_data"]
        
        # ─────────────────────────────────────────────────────────────
        # Step 4: Async Logging
        # ─────────────────────────────────────────────────────────────
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Log asynchronously to avoid blocking response
        background_tasks.add_task(
            logger_service.log_request,
            request=request.dict(),
            response=response,
            status="success",
            errors=[],
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"Forecast generated successfully | "
            f"processing_time_ms={processing_time_ms}"
        )
        
        # ─────────────────────────────────────────────────────────────
        # Step 5: Return Response
        # ─────────────────────────────────────────────────────────────
        
        return JSONResponse(status_code=200, content=response)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(
            f"Forecast generation failed: {str(e)}",
            exc_info=True,
            extra={"company": request.company}
        )
        
        # ─────────────────────────────────────────────────────────────
        # Error Response
        # ─────────────────────────────────────────────────────────────
        
        error_response = {
            "forecast_summary": "",
            "key_financial_trends": [],
            "management_outlook": "",
            "risks_and_opportunities": [],
            "source_documents": [],
            "errors": [f"Forecast generation failed: {str(e)}"]
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
        
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all error handler for unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ─────────────────────────────────────────────────────────────
# Application Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )


# ─────────────────────────────────────────────────────────────
# File: app/config.py
# Configuration management
# ─────────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"
    log_level: str = "INFO"
    
    # LLM Configuration
    openai_api_key: str
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str = "tcs-financial-docs"
    pinecone_environment: str = "gcp-starter"
    
    # MySQL Configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str
    mysql_database: str = "tcs_forecast"
    
    # Feature Flags
    include_market_data: bool = False
    enable_hallucination_guard: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# ─────────────────────────────────────────────────────────────
# File: app/models/forecast.py
# Request/Response models
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""
    company: str
    include_market_data: Optional[bool] = False

class MarketData(BaseModel):
    """Market data structure."""
    stock_price: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    retrieved_at: str

class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    forecast_summary: str
    key_financial_trends: List[str]
    management_outlook: str
    risks_and_opportunities: List[str]
    market_data: Optional[MarketData] = None
    source_documents: List[str]
    errors: List[str]


# ─────────────────────────────────────────────────────────────
# File: app/services/logger.py
# Async database logging
# ─────────────────────────────────────────────────────────────

import aiomysql
import json
from typing import Any, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AsyncDatabaseLogger:
    """Async MySQL logger for API requests and responses."""
    
    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool
    
    async def log_request(self,
                          request: Dict[str, Any],
                          response: Dict[str, Any],
                          status: str,
                          errors: List[str],
                          user_id: Optional[str] = None,
                          processing_time_ms: Optional[int] = None):
        """
        Log API request and response to MySQL.
        
        Args:
            request: Request payload
            response: Response payload
            status: 'success' or 'error'
            errors: List of error messages
            user_id: Optional user identifier
            processing_time_ms: Time to process request
        """
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    query = """
                        INSERT INTO api_request_logs
                        (request_payload, response_payload, status, 
                         error_messages, user_id, processing_time_ms)
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
                    
                    logger.debug(
                        f"Logged {status} request | "
                        f"processing_time_ms={processing_time_ms}"
                    )
        
        except Exception as e:
            # Logging failure should not crash the application
            logger.error(
                f"Failed to log request to database: {str(e)}",
                exc_info=True
            )
            # In production, could queue for retry


# ─────────────────────────────────────────────────────────────
# Usage & Testing
# ─────────────────────────────────────────────────────────────

"""
Quick Start:

1. Install dependencies:
   pip install -r requirements.txt

2. Create .env file:
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   MYSQL_PASSWORD=...
   etc.

3. Initialize database:
   mysql -u root -p < database/init.sql

4. Run application:
   python app/main.py
   
   OR:
   uvicorn app.main:app --reload

5. Test endpoint:
   curl -X POST http://localhost:8000/forecast \\
     -H "Content-Type: application/json" \\
     -d '{"company": "TCS", "include_market_data": true}'

6. View API docs:
   http://localhost:8000/docs (Swagger UI)
   http://localhost:8000/redoc (ReDoc)
"""
