# VoiceFlow-Dispatcher Agent

## 1. Executive Summary
VoiceFlow-Dispatcher is a **real-time, asynchronous voice dispatcher** designed to automate field service workflows. 

**Key Capabilities:**
* **Real-time STT/TTS:** .
* **Asynchronous Processing:** Uses Azure Service Bus to decouple ingestion from inference, ensuring high availability during call spikes.
* **Active Learning (RL):** Implements a Contextual Bandit algorithm to optimize response verbosity based on interaction history.

## 2. Architecture

**The Flow:**
1.  **Ingest Service (Edge):** Captures audio -> Azure Speech SDK -> Text.
2.  **Event Broker:** Pushes raw text events to **Azure Service Bus**.
3.  **Inference Worker:**
    * Pulls message.
    * **RAG/NLU:** specifices Intent via **Azure OpenAI** using Function Calling.
    * **RL Layer:** Selects response strategy (Concise vs. Detailed).
    * **Action:** Executes SQL transaction in **PostgreSQL**.
4.  **Response:** Generates Azure Neural TTS audio.


## 3. Env example
```
AZURE_OPENAI_KEY=...
AZURE_OPENAI_MODEL=...
AZURE_OPENAI_ENDPOINT=...
AZURE_SERVICEBUS_CONN_STR=...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=...
SERVICEBUS_QUEUE_NAME=...
POSTGRES_URI=...
```

> **_NOTE:_** for the best experent please use *gpt-4.1* models (mini, nano ,etc.)

