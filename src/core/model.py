import uuid
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Job/Ticket Domain Models
# ============================================================================

class JobStatus(str, Enum):
    """Status of a field service job"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Field service job/ticket model"""
    job_id: str = Field(default_factory=lambda: f"JOB-{uuid.uuid4().hex[:8]}")
    customer_name: str
    location: str | None = None
    description: str | None = None
    status: JobStatus = JobStatus.PENDING
    assigned_technician: str | None = None
    parts_used: list[str] = Field(default_factory=list)
    billing_hours: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


# ============================================================================
# Intent/Action Models
# ============================================================================

class IntentType(str, Enum):
    """Type of intent extracted from voice input"""
    CREATE_JOB = "create_job"
    UPDATE_JOB = "update_job"
    CLOSE_JOB = "close_job"
    QUERY_JOB = "query_job"
    LIST_JOBS = "list_jobs"
    ADD_NOTES = "add_notes"
    UNKNOWN = "unknown"


class ExtractedIntent(BaseModel):
    """Structured data extracted from voice input"""
    intent: IntentType
    customer: str | None = None
    action: str | None = None
    parts: list[str] = Field(default_factory=list)
    billing_hours: float | None = None
    job_id: str | None = None
    notes: str | None = None
    confidence: float = 1.0
    raw_text: str

    model_config = ConfigDict(frozen=True)


# ============================================================================
# Message/Event Models
# ============================================================================

class MessageType(str, Enum):
    """Type of message in the system"""
    VOICE_INPUT = "voice_input"
    INTENT_EXTRACTED = "intent_extracted"
    ACTION_COMPLETED = "action_completed"
    ERROR = "error"


class VoiceMessage(BaseModel):
    """Message envelope for Service Bus"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]
    technician_id: str
    session_id: str | None = None

    model_config = ConfigDict(frozen=True)


# ============================================================================
# RL Context Models
# ============================================================================

class ResponseStyle(str, Enum):
    """Style of response to deliver to user"""
    CONCISE = "concise"      # "Got it. Job closed."
    DETAILED = "detailed"    # "Job #104 closed. 2 hours billed. Next ticket?"
    VERBOSE = "verbose"      # Full details with confirmation


class UserContext(BaseModel):
    """Context for RL decision making"""
    technician_id: str
    time_of_day: int = Field(ge=0, le=23)  # Hour of day
    interaction_count: int = 0
    recent_errors: int = 0
    preferred_style: ResponseStyle | None = None
    avg_response_time: float = 0.0

    model_config = ConfigDict(frozen=False)


# ============================================================================
# In-Memory Repository
# ============================================================================

class JobRepository:
    """Simple in-memory job storage"""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job: Job) -> Job:
        """Create a new job"""
        with self._lock:
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        """Get job by ID"""
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job) -> Job:
        """Update existing job"""
        with self._lock:
            if job.job_id not in self._jobs:
                raise KeyError(f"Job {job.job_id} not found")
            job.updated_at = datetime.now(timezone.utc)
            self._jobs[job.job_id] = job
            return job

    def delete(self, job_id: str) -> None:
        """Delete job by ID"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]

    def list_by_technician(self, technician_id: str) -> list[Job]:
        """List all jobs for a technician"""
        with self._lock:
            return [
                job for job in self._jobs.values()
                if job.assigned_technician == technician_id
            ]

    def list_by_status(self, status: JobStatus) -> list[Job]:
        """List all jobs with given status"""
        with self._lock:
            return [
                job for job in self._jobs.values()
                if job.status == status
            ]

    def list_all(self) -> list[Job]:
        """List all jobs"""
        with self._lock:
            return list(self._jobs.values())

    def count(self) -> int:
        """Count total jobs"""
        with self._lock:
            return len(self._jobs)
