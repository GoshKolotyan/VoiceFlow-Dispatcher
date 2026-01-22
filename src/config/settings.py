from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#For future move into src/core/models
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Azure OpenAI Configuration
    AZURE_OPENAI_KEY: str = Field(validation_alias='AZURE_OPENAI_KEY')
    AZURE_OPENAI_ENDPOINT: str = Field(validation_alias='AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_MODEL: str = Field(default="gpt-4o", validation_alias='AZURE_OPENAI_MODEL')

    # Azure Speech Configuration
    AZURE_SPEECH_REGION: str = Field(validation_alias='AZURE_SPEECH_REGION')
    AZURE_SPEECH_KEY: str = Field(validation_alias='AZURE_SPEECH_KEY')

    # Azure Service Bus Configuration
    AZURE_SERVICEBUS_CONN_STR: str = Field(validation_alias='AZURE_SERVICEBUS_CONN_STR')
    SERVICEBUS_QUEUE_NAME: str = Field(default="voice-input-queue")

    # PostgreSQL Configuration (for future use)
    POSTGRES_URI: str = Field(validation_alias='POSTGRES_URI')

    # RL Parameters
    RL_EPSILON: float = 0.1

    # Application Configuration
    LOG_LEVEL: str = Field(default="INFO")
    MAX_RETRIES: int = Field(default=3)
    TIMEOUT_SECONDS: int = Field(default=30)

# if __name__ == "__main__":
#     settings = Settings()# type: ignore
#     print(settings.model_dump())