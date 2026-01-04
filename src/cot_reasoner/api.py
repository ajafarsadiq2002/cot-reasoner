"""REST API for the Chain of Thought Reasoner using FastAPI."""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
import typer
import uvicorn

# Load environment variables from .env file
load_dotenv()
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cot_reasoner import __version__
from cot_reasoner.core.reasoner import Reasoner
from cot_reasoner.db import get_db, init_db

# FastAPI app
api = FastAPI(
    title="CoT Reasoner API",
    description="Chain of Thought Reasoning API with multi-provider LLM support",
    version=__version__,
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


# Pydantic model for stats
class StatsResponse(BaseModel):
    """Database statistics response."""

    total: int
    completed: int
    failed: int
    pending: int
    avg_tokens: float


# Pydantic models for request/response
class ReasonRequest(BaseModel):
    """Request model for reasoning endpoint."""

    query: str = Field(..., description="The problem or question to reason about")
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic)")
    model: Optional[str] = Field(default=None, description="Model to use")
    strategy: str = Field(default="standard", description="Reasoning strategy")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum tokens")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is 15% of 240?",
                    "provider": "openai",
                    "strategy": "standard",
                }
            ]
        }
    }


class ReasoningStepResponse(BaseModel):
    """Response model for a reasoning step."""

    number: int
    content: str
    confidence: float


class ReasonResponse(BaseModel):
    """Response model for reasoning result."""

    id: str
    query: str
    steps: list[ReasoningStepResponse]
    answer: Optional[str]
    confidence: float
    provider: str
    model: str
    strategy: str
    total_tokens: int
    created_at: str
    status: str = "completed"


class ReasonAsyncResponse(BaseModel):
    """Response for async reasoning request."""

    id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class ProvidersResponse(BaseModel):
    """Available providers response."""

    providers: list[str]


class StrategiesResponse(BaseModel):
    """Available strategies response."""

    strategies: list[str]


# API Endpoints
@api.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat(),
    )


@api.get("/providers", response_model=ProvidersResponse, tags=["System"])
async def list_providers():
    """List available LLM providers."""
    return ProvidersResponse(providers=Reasoner.list_providers())


@api.get("/strategies", response_model=StrategiesResponse, tags=["System"])
async def list_strategies():
    """List available reasoning strategies."""
    return StrategiesResponse(strategies=Reasoner.list_strategies())


@api.post("/reason", response_model=ReasonResponse, tags=["Reasoning"])
async def reason(request: ReasonRequest):
    """
    Perform Chain of Thought reasoning on a query.

    This endpoint processes the query synchronously and returns the complete
    reasoning chain with steps and final answer.
    """
    try:
        reasoner = Reasoner(
            provider=request.provider,
            model=request.model,
            strategy=request.strategy,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Use async reasoning
        result = await reasoner.reason_async(request.query)

        # Generate unique ID
        result_id = str(uuid.uuid4())

        # Save to database
        db = get_db()
        db.save_result(result_id, result, status="completed")

        # Build response
        response = ReasonResponse(
            id=result_id,
            query=result.query,
            steps=[
                ReasoningStepResponse(
                    number=step.number,
                    content=step.content,
                    confidence=step.confidence,
                )
                for step in result.steps
            ],
            answer=result.answer,
            confidence=result.confidence,
            provider=result.provider,
            model=result.model,
            strategy=result.strategy,
            total_tokens=result.total_tokens,
            created_at=result.created_at.isoformat(),
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")


@api.post("/reason/async", response_model=ReasonAsyncResponse, tags=["Reasoning"])
async def reason_async(request: ReasonRequest, background_tasks: BackgroundTasks):
    """
    Submit a reasoning request for async processing.

    Returns immediately with a task ID. Use GET /reason/{id} to retrieve results.
    """
    task_id = str(uuid.uuid4())
    db = get_db()

    # Store pending status in database
    db.save_result(task_id, query=request.query, status="pending")

    async def process_reasoning():
        try:
            reasoner = Reasoner(
                provider=request.provider,
                model=request.model,
                strategy=request.strategy,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            result = await reasoner.reason_async(request.query)

            # Update database with completed result
            db.save_result(task_id, result, status="completed")

        except Exception as e:
            # Update database with failed status
            db.save_result(task_id, query=request.query, status="failed", error=str(e))

    background_tasks.add_task(process_reasoning)

    return ReasonAsyncResponse(
        id=task_id,
        status="pending",
        message="Reasoning task submitted. Use GET /reason/{id} to retrieve results.",
    )


@api.get("/reason/{task_id}", tags=["Reasoning"])
async def get_reasoning_result(task_id: str):
    """
    Retrieve a reasoning result by ID.

    Works for both sync and async reasoning requests.
    """
    db = get_db()
    result = db.get_result(task_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Reasoning result not found")

    return result


@api.get("/results", tags=["Reasoning"])
async def list_results(limit: int = 10, status: Optional[str] = None):
    """
    List recent reasoning results.

    Args:
        limit: Maximum number of results (default: 10)
        status: Filter by status (pending, completed, failed)
    """
    db = get_db()
    return db.get_recent_results(limit=limit, status=status)


@api.delete("/reason/{task_id}", tags=["Reasoning"])
async def delete_reasoning_result(task_id: str):
    """Delete a reasoning result by ID."""
    db = get_db()
    deleted = db.delete_result(task_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Reasoning result not found")

    return {"message": "Result deleted", "id": task_id}


@api.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get database statistics."""
    db = get_db()
    return db.get_stats()


@api.post("/reason/stream", tags=["Reasoning"])
async def reason_stream(request: ReasonRequest):
    """
    Stream the reasoning process.

    Returns a server-sent events stream of the reasoning as it's generated.
    """
    try:
        reasoner = Reasoner(
            provider=request.provider,
            model=request.model,
            strategy=request.strategy,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        async def generate():
            # Note: This uses sync streaming wrapped in async
            # For production, implement proper async streaming in providers
            for chunk in reasoner.reason_stream(request.query):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


# CLI for running the API server
cli = typer.Typer()


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
):
    """Start the CoT Reasoner API server."""
    print(f"Starting CoT Reasoner API on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")

    uvicorn.run(
        "cot_reasoner.api:api",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


def main():
    """Entry point for the API CLI."""
    cli()


if __name__ == "__main__":
    main()
