"""Pydantic schemas for the FastAPI prediction endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FlightInput(BaseModel):
    month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=1, le=7)
    dep_hour: int = Field(..., ge=0, le=23)
    carrier: str = Field(..., min_length=2, max_length=3)
    origin: str = Field(..., min_length=3, max_length=4)
    dest: str = Field(..., min_length=3, max_length=4)
    distance: float = Field(..., ge=0, le=15000)
    crs_elapsed_time: float = Field(..., ge=0, le=1440)

    @field_validator("carrier", "origin", "dest", mode="before")
    @classmethod
    def uppercase_strings(cls, v: str) -> str:
        return v.strip().upper()


class BatchFlightInput(BaseModel):
    flights: list[FlightInput] = Field(..., min_length=1, max_length=100)


class PredictionResult(BaseModel):
    delay_probability: float
    is_delayed: bool
    confidence: str
    threshold_used: float = 0.5


class PredictionResponse(BaseModel):
    input: FlightInput
    prediction: PredictionResult
    model_version: Optional[str] = None
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    model_version: Optional[str] = None
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
