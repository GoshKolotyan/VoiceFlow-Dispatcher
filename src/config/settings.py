from pydantic import BaseModel


class Settings(BaseModel):
    AZURE_SPEECH_KEY: str
    AZURE_SPEECH_REGION: str