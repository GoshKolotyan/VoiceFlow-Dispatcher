import json
from typing import Any
from openai import AsyncAzureOpenAI

from src.core.logger import LoggerFactory
from src.core.model import ExtractedIntent, IntentType, ResponseStyle
from src.core.expections import OpenAIServiceError
from src.config.settings import Settings


class AzureOpenAIService:
    """Azure OpenAI service with function calling for intent extraction"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = LoggerFactory.create_logger("AzureOpenAI", level=settings.LOG_LEVEL)

        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_KEY,
            api_version="2024-08-01-preview",
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )

        self.model = settings.AZURE_OPENAI_MODEL
        self.tools = self._define_function_tools()

        self.logger.info(f"Initialized Azure OpenAI service with model {self.model}")

    def _define_function_tools(self) -> list[dict[str, Any]]:
        """Define function calling schema for intent extraction"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_field_service_intent",
                    "description": "Extract structured information from technician voice input about field service jobs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "create_job",
                                    "update_job",
                                    "close_job",
                                    "query_job",
                                    "list_jobs",
                                    "add_notes"
                                ],
                                "description": "The primary intent or action the technician wants to perform"
                            },
                            "customer": {
                                "type": "string",
                                "description": "Customer name or identifier mentioned in the input"
                            },
                            "action": {
                                "type": "string",
                                "description": "Specific action requested (e.g., 'close_ticket', 'add_parts', 'update_status')"
                            },
                            "parts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of parts or equipment used or mentioned"
                            },
                            "billing_hours": {
                                "type": "number",
                                "description": "Number of hours to bill for the service"
                            },
                            "job_id": {
                                "type": "string",
                                "description": "Specific job ID if mentioned"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Additional notes, descriptions, or context provided"
                            }
                        },
                        "required": ["intent"]
                    }
                }
            }
        ]

    async def extract_intent(
        self,
        text: str,
        context: dict[str, Any] | None = None
    ) -> ExtractedIntent:
        """Extract structured intent from natural language text using function calling"""
        self.logger.info(f"Extracting intent from: '{text[:50]}...'")

        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an AI assistant for field service technicians.
Extract structured information from their voice commands about service jobs.
Focus on: job updates, customer names, parts used, billing hours, and any notes.
Be lenient with informal language and handle messy speech-to-text transcriptions."""
                },
                {"role": "user", "content": text}
            ]

            # Add context if provided
            if context:
                messages[0]["content"] += f"\n\nAdditional context: {json.dumps(context)}"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="required",
                max_completion_tokens=500
            )

            message = response.choices[0].message

            # Parse tool call response
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)

                intent = ExtractedIntent(
                    intent=IntentType(arguments.get("intent", "unknown")),
                    customer=arguments.get("customer"),
                    action=arguments.get("action"),
                    parts=arguments.get("parts") or [],
                    billing_hours=arguments.get("billing_hours"),
                    job_id=arguments.get("job_id"),
                    notes=arguments.get("notes"),
                    raw_text=text,
                    confidence=0.9  # High confidence if function call succeeded
                )

                self.logger.info(
                    f"Extracted intent: {intent.intent}, customer: {intent.customer}"
                )
                return intent
            else:
                # Fallback if no tool call (shouldn't happen often)
                self.logger.warning("No function call in response, returning unknown intent")
                return ExtractedIntent(
                    intent=IntentType.UNKNOWN,
                    raw_text=text,
                    confidence=0.3
                )

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse function call arguments: {e}")
            raise OpenAIServiceError(
                "Invalid JSON in function call response",
                details={"text": text, "error": str(e)}
            )
        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            raise OpenAIServiceError(
                "Failed to extract intent from text",
                details={"text": text, "error": str(e), "type": type(e).__name__}
            )

    async def generate_response(
        self,
        intent: ExtractedIntent,
        context: dict[str, Any],
        style: ResponseStyle = ResponseStyle.DETAILED
    ) -> str:
        """Generate natural language response based on intent, context, and style"""
        self.logger.info(f"Generating {style} response for intent: {intent.intent}")

        # Define style instructions
        style_instructions = {
            ResponseStyle.CONCISE: "Respond in 5 words or less. Be brief and direct.",
            ResponseStyle.DETAILED: "Provide a clear, one-sentence confirmation with key details.",
            ResponseStyle.VERBOSE: "Provide full details, confirm the action, and ask about next steps."
        }

        # Build context summary
        context_summary = json.dumps(context, indent=2)

        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a field service assistant providing feedback to technicians.
{style_instructions[style]}

Guidelines:
- Be professional but friendly
- Confirm what was done
- Include relevant details (job IDs, billing hours, etc.)
- Use natural conversational language
- Keep responses focused and actionable"""
                },
                {
                    "role": "user",
                    "content": f"""The technician said: "{intent.raw_text}"

We interpreted this as: {intent.intent}
Action result: {context_summary}

Generate an appropriate response."""
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=150
            )

            response_text = response.choices[0].message.content or ""

            # If model returns empty response, use fallback
            if not response_text.strip():
                self.logger.warning("Model returned empty response, using fallback")
                return self._generate_fallback_response(intent, context, style)

            self.logger.info(f"Generated response: '{response_text[:50]}...'")
            return response_text

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            # Fallback to simple confirmation
            fallback = self._generate_fallback_response(intent, context, style)
            self.logger.warning(f"Using fallback response: {fallback}")
            return fallback

    def _generate_fallback_response(
        self,
        intent: ExtractedIntent,
        context: dict[str, Any],
        style: ResponseStyle
    ) -> str:
        """Generate simple fallback response if API fails"""
        success = context.get("success", False)

        if not success:
            return "I encountered an error processing your request. Please try again."

        # Simple template-based responses
        if style == ResponseStyle.CONCISE:
            return "Done."
        elif style == ResponseStyle.DETAILED:
            if intent.intent == IntentType.CLOSE_JOB:
                job_id = context.get("job_id", "unknown")
                return f"Job {job_id} closed successfully."
            elif intent.intent == IntentType.CREATE_JOB:
                job_id = context.get("job_id", "unknown")
                return f"Created job {job_id}."
            else:
                return "Request processed successfully."
        else:  # VERBOSE
            message = context.get("message", "Action completed")
            return f"{message}. Is there anything else you need?"

    def __repr__(self) -> str:
        return f"AzureOpenAIService(model={self.model})"
