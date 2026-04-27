from __future__ import annotations

from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from candidate_qa import answer_candidate_question, build_candidate_resume_context
from ingest_resume import create_candidate_and_resume
from serch_candidates import CandidateSearchResult, search_candidates


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="HR-запрос на поиск кандидатов")
    top_k: int = Field(default=20, ge=1, le=100)


class CandidateCreateRequest(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: str = ""
    phone: str = ""
    role: str = ""
    raw_resume_text: str = Field(..., min_length=1)


class CandidateQuestionMessage(BaseModel):
    role: str
    content: str


class CandidateQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[CandidateQuestionMessage] = Field(default_factory=list)


class AssistantPromptRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = "chat"


app = FastAPI(
    title="HR Platform API",
    version="1.0.0",
    description="HTTP API для фронтенда дипломного проекта интеллектуальной HR-платформы.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "message": detail.get("message", "Request failed"),
                "details": detail.get("details"),
            },
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": str(detail),
            "details": None,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "message": str(exc) or "Internal Server Error",
            "details": None,
        },
    )


def _search_result_to_dict(result: CandidateSearchResult) -> Dict[str, Any]:
    return {
        "raw_query": result.raw_query,
        "normalized_query": result.normalized_query,
        "is_hr_relevant": result.is_hr_relevant,
        "should_search_candidates": result.should_search_candidates,
        "short_explanation": result.short_explanation,
        "request_type": result.request_type,
        "intent": result.intent,
        "confidence": result.confidence,
        "candidates": result.candidates,
    }


def _map_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    score = candidate.get("total_score")
    role = candidate.get("role", "") or ""
    email = candidate.get("email", "") or ""
    meta_parts = [part for part in [role, email] if part]

    return {
        "id": str(candidate["candidate_id"]),
        "fullName": candidate.get("full_name", ""),
        "headline": role or None,
        "location": None,
        "score": score,
        "meta": " • ".join(meta_parts) if meta_parts else "Кандидат из базы",
        "tags": [role] if role else [],
    }


def _format_assistant_answer(result: CandidateSearchResult) -> str:
    if not result.is_hr_relevant or not result.should_search_candidates:
        return result.short_explanation or "Запрос не похож на поиск кандидата."

    if not result.candidates:
        return "По этому запросу кандидаты не найдены."

    lines = ["Нашел кандидатов:"]
    for idx, candidate in enumerate(result.candidates[:5], start=1):
        role = candidate.get("role", "") or "роль не указана"
        score = candidate.get("total_score", 0.0)
        lines.append(
            f"{idx}. {candidate.get('full_name', 'Без имени')} — {role} (score {score:.2f})"
        )
    return "\n".join(lines)


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@api_router.get("/health")
def api_healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@api_router.get("/candidates/search")
def candidates_search(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
) -> Dict[str, Any]:
    try:
        result = search_candidates(query, top_k=limit)
        items = [_map_candidate(candidate) for candidate in result.candidates[:limit]]
        return {
            "items": items,
            "total": len(items),
            "meta": _search_result_to_dict(result),
        }
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Database error", "details": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "details": None},
        ) from exc


@api_router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int) -> Dict[str, Any]:
    context = build_candidate_resume_context(candidate_id)
    if context is None:
        raise HTTPException(status_code=404, detail={"message": "Candidate not found"})

    return {
        "candidate": context["meta"],
        "resume_chunks": context["chunks"],
    }


@api_router.post("/candidates")
def create_candidate(payload: CandidateCreateRequest) -> Dict[str, Any]:
    try:
        creation_result = create_candidate_and_resume(
            full_name=payload.full_name,
            email=payload.email,
            phone=payload.phone,
            role=payload.role,
            raw_resume_text=payload.raw_resume_text,
        )
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Database error", "details": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "details": None},
        ) from exc

    return {
        "message": "Candidate created and indexed successfully",
        "candidate": {
            "candidate_id": creation_result["candidate_id"],
            "resume_id": creation_result["resume_id"],
            "full_name": payload.full_name,
            "email": payload.email,
            "phone": payload.phone,
            "role": payload.role,
            "chunks_count": creation_result["chunks_count"],
        },
    }


@api_router.post("/search")
def search(payload: SearchRequest) -> Dict[str, Any]:
    try:
        result = search_candidates(payload.query, top_k=payload.top_k)
        return _search_result_to_dict(result)
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Database error", "details": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "details": None},
        ) from exc


@api_router.post("/candidates/{candidate_id}/qa")
def candidate_qa(candidate_id: int, payload: CandidateQuestionRequest) -> Dict[str, Any]:
    try:
        result = answer_candidate_question(
            candidate_id=candidate_id,
            question=payload.question,
            history=[item.model_dump() for item in payload.history],
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"message": str(exc), "details": None},
        ) from exc
    except psycopg2.Error as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Database error", "details": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "details": None},
        ) from exc


@api_router.post("/assistant/messages")
def assistant_messages(payload: AssistantPromptRequest) -> Dict[str, Any]:
    try:
        result = search_candidates(payload.message, top_k=5)
        return {
            "answer": _format_assistant_answer(result),
            "conversationId": None,
            "search": _search_result_to_dict(result),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": str(exc), "details": None},
        ) from exc


@api_router.post("/assistant/stt")
async def assistant_stt(file: Optional[UploadFile] = None) -> Dict[str, Any]:
    if file is None:
        return {"text": ""}

    return {"text": ""}


app.include_router(api_router)
