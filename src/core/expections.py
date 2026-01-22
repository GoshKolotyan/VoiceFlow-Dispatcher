from datetime import datetime, timezone
from typing import Any


# ============================================================================
# Base Exception
# ============================================================================

class VoiceFlowException(Exception):
    """Base exception for VoiceFlow-Dispatcher"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to dictionary"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# Azure Service Exceptions
# ============================================================================

class AzureServiceException(VoiceFlowException):
    """Base exception for Azure service errors"""
    pass


class SpeechRecognitionError(AzureServiceException):
    """Speech-to-text recognition failed"""
    pass


class SpeechSynthesisError(AzureServiceException):
    """Text-to-speech synthesis failed"""
    pass


class OpenAIServiceError(AzureServiceException):
    """Azure OpenAI API errors"""
    pass


class ServiceBusError(AzureServiceException):
    """Azure Service Bus messaging errors"""
    pass


# ============================================================================
# Domain Exceptions
# ============================================================================

class JobNotFoundError(VoiceFlowException):
    """Job lookup failed - job does not exist"""
    pass


class IntentExtractionError(VoiceFlowException):
    """Failed to extract intent from voice input"""
    pass


class InvalidJobStateError(VoiceFlowException):
    """Invalid job state transition attempted"""
    pass


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(VoiceFlowException):
    """Configuration or settings error"""
    pass


class AuthenticationError(VoiceFlowException):
    """Authentication failed with Azure services"""
    pass
