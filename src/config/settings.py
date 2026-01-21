from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#For future move into src/core/models
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    AZURE_OPENAI_KEY: str = Field(validation_alias='AZURE_OPENAI_KEY') 
    AZURE_SPEECH_REGION: str = Field(validation_alias='AZURE_SPEECH_REGION')
    AZURE_SPEECH_KEY: str = Field(validation_alias='AZURE_SPEECH_KEY')
    AZURE_SERVICEBUS_CONN_STR: str = Field(validation_alias='AZURE_SERVICEBUS_CONN_STR')
    POSTGRES_URI: str = Field(validation_alias='POSTGRES_URI')
    
    # RL Parameters
    RL_EPSILON: float = 0.1

# if __name__ == "__main__":
#     settings = Settings()# type: ignore
#     print(settings.model_dump())