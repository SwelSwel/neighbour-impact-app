"""Request and response models. Pydantic rejects invalid input with HTTP 422 before it reaches the model.

A KPI value may be null: it then becomes a missing value, exactly like a missing reading in the notebook
(Gradient Boosting handles missing values natively).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonNeg = Annotated[float, Field(ge=0)]
Positive = Annotated[float, Field(gt=0)]
Percent = Annotated[float, Field(ge=0, le=100)]
NonNegInt = Annotated[int, Field(ge=0)]


# ----------------------------------------------------------------------------- inputs
class KPIReading(BaseModel):
    """One 3-hourly KPI record of a cell: the 8 raw KPIs in Dataset 2. Null = not recorded."""
    model_config = ConfigDict(extra="forbid")
    Traffic: NonNeg | None = None
    Connected_Users: NonNeg | None = None
    Throughput: NonNeg | None = None
    Latency: NonNeg | None = Field(default=None, description="ms")
    Packet_Loss: Percent | None = Field(default=None, description="%")
    Alarm_Count: NonNeg | None = None
    Load: NonNeg | None = None
    Capacity: Positive | None = None


class LastWeek(BaseModel):
    """The neighbour's values at the predicted slot one week earlier (optional)."""
    model_config = ConfigDict(extra="forbid")
    Latency: NonNeg | None = None
    Utilization: NonNeg | None = None
    Packet_Loss: Percent | None = None
    Traffic: NonNeg | None = None
    Load: NonNeg | None = None


class FailedCell(BaseModel):
    """The failed cell at its last reading before the outage."""
    model_config = ConfigDict(extra="forbid")
    cell_id: str = Field(min_length=1, max_length=64)
    Traffic: NonNeg | None = None
    Connected_Users: NonNeg | None = None
    Load: NonNeg | None = None
    Capacity: Positive | None = None
    Throughput: NonNeg | None = None
    site_cells_down: int = Field(default=0, ge=0, le=50,
                                 description="Other cells on the same site already down when this outage starts")


class Neighbour(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cell_id: str = Field(min_length=1, max_length=64)
    is_same_site: bool
    is_listed: bool
    now: KPIReading = Field(description="Last reading before the outage")
    h3_before: KPIReading | None = Field(default=None, description="Reading 3 hours before that")
    h6_before: KPIReading | None = Field(default=None, description="Reading 6 hours before that")
    last_week: LastWeek | None = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outage_start: datetime
    failed_cell: FailedCell
    neighbours: list[Neighbour] = Field(min_length=1, description="Every neighbour that was up just before the outage")

    @field_validator("neighbours")
    @classmethod
    def unique_ids(cls, v: list[Neighbour]) -> list[Neighbour]:
        ids = [n.cell_id for n in v]
        if len(ids) != len(set(ids)):
            raise ValueError("neighbour cell_id values must be unique")
        return v

    @model_validator(mode="after")
    def failed_not_neighbour(self):
        if any(n.cell_id == self.failed_cell.cell_id for n in self.neighbours):
            raise ValueError("the failed cell cannot also be listed as its own neighbour")
        return self


class CellTimeRequest(BaseModel):
    """Look up (and predict) an outage from the data store: only the cell and the start time are needed."""
    model_config = ConfigDict(extra="forbid")
    cell_id: str = Field(min_length=1, max_length=64)
    outage_start: datetime


# ----------------------------------------------------------------------------- outputs
class KPIForecast(BaseModel):
    before: float | None = Field(description="Value at the last reading before the outage")
    predicted: float
    range_80_low: float
    range_80_high: float
    change_pct: float | None
    degraded: bool
    severe: bool
    actual: float | None = Field(default=None, description="Observed value, when known (looked-up outages only)")


class NeighbourPrediction(BaseModel):
    cell_id: str
    is_same_site: bool
    is_listed: bool
    load_ratio: float | None
    Latency: KPIForecast
    Utilization: KPIForecast
    Packet_Loss: KPIForecast
    any_degraded: bool
    any_severe: bool


class PredictResponse(BaseModel):
    request_id: str
    failed_cell: str
    prediction_time: datetime = Field(description="The scheduled 3-hourly reading the predictions refer to")
    model: str
    app_version: str
    degraded_threshold_pct: float
    severe_threshold_pct: float
    n_neighbours: int
    n_degraded: int
    neighbours: list[NeighbourPrediction]


class SkippedNeighbour(BaseModel):
    cell_id: str
    reason: str


class LookupResponse(BaseModel):
    """Everything the app found for one outage. `request` can be edited and sent to POST /predict."""
    request: PredictRequest
    prediction_time: datetime
    actual: dict[str, dict[str, float | None] | None] = Field(
        description="Observed Latency, Utilization, Packet_Loss per neighbour at the predicted slot (null if unknown)")
    real_outage: dict | None = Field(description="The recorded outage this lookup matches, if any")
    n_neighbours_total: int
    skipped: list[SkippedNeighbour] = Field(description="Neighbours left out because they were down or not reporting")
    source: str = Field(description="'data store' or 'upload'")


class PredictByCellResponse(BaseModel):
    lookup: LookupResponse
    prediction: PredictResponse


class OutageItem(BaseModel):
    event_id: int | None
    cell_id: str
    start: datetime
    up: datetime | None
    duration_h: float | None
    split: str
    technology: str | None = None
    region: str | None = None
    n_neighbours: int | None = None
