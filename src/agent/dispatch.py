import time
import uuid
from datetime import datetime, timezone

from src.core.logger import LoggerFactory
from src.core.model import (
    Job, JobStatus, JobRepository,
    IntentType, UserContext, ResponseStyle,
    VoiceMessage, MessageType
)
from src.core.expections import JobNotFoundError, VoiceFlowException
from src.config.settings import Settings
from src.services.azure_speech import AzureSpeechService
from src.services.azure_openai import AzureOpenAIService
from src.services.azure_bus import AzureServiceBusProducer, AzureServiceBusConsumer
from src.agent.rl_bandit import ContextualBandit


class DispatchAgent:
    """Main dispatcher agent orchestrating the voice workflow"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = LoggerFactory.create_logger("DispatchAgent", level=settings.LOG_LEVEL)

        # Initialize services
        self.logger.info("Initializing services...")
        self.speech_service = AzureSpeechService(settings)
        self.openai_service = AzureOpenAIService(settings)
        self.bus_producer = AzureServiceBusProducer(settings)
        self.rl_bandit = ContextualBandit(epsilon=settings.RL_EPSILON, log_level=settings.LOG_LEVEL)

        # In-memory storage
        self.job_repo = JobRepository()
        self.user_contexts: dict[str, UserContext] = {}

        # Session tracking
        self.current_session_id: str | None = None

        self.logger.info("DispatchAgent initialized successfully")

    async def handle_voice_input(self, technician_id: str) -> str:
        """
        Main entry point: handle voice input from technician

        Args:
            technician_id: Identifier for the technician

        Returns:
            Response text that was spoken back to technician

        Raises:
            VoiceFlowException: If processing fails
        """
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        start_time = time.time()

        self.logger.info(f"=== Starting voice interaction for {technician_id} (session: {session_id}) ===")

        try:
            # Step 1: Speech to Text
            self.logger.info("Step 1: Listening for voice input...")
            text = await self.speech_service.speech_to_text()

            # Step 2: Send to Service Bus (async buffer)
            self.logger.info("Step 2: Sending message to Service Bus...")
            message = VoiceMessage(
                message_type=MessageType.VOICE_INPUT,
                payload={"text": text},
                technician_id=technician_id,
                session_id=session_id
            )
            await self.bus_producer.send_message(message)

            # Step 3: Process message (simulate worker pulling from queue)
            self.logger.info("Step 3: Processing message...")
            response_text = await self._process_message(message)

            # Step 4: Text to Speech
            self.logger.info("Step 4: Speaking response...")
            await self.speech_service.text_to_speech(response_text)

            # Step 5: Update user context and calculate reward
            response_time = time.time() - start_time
            self._update_user_context(
                technician_id=technician_id,
                response_time=response_time,
                error_occurred=False,
                user_repeated=False
            )

            self.logger.info(
                f"=== Interaction completed successfully in {response_time:.2f}s ==="
            )
            return response_text

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Voice interaction failed: {e}")

            # Generate error response
            error_response = "I'm sorry, I encountered an error. Please try again."

            try:
                await self.speech_service.text_to_speech(error_response)
            except Exception as tts_error:
                self.logger.error(f"Failed to speak error message: {tts_error}")

            # Update context with error
            self._update_user_context(
                technician_id=technician_id,
                response_time=response_time,
                error_occurred=True,
                user_repeated=False
            )

            self.logger.error(
                f"=== Interaction failed after {response_time:.2f}s ==="
            )
            return error_response

    async def _process_message(self, message: VoiceMessage) -> str:
        """
        Worker logic: process message from queue

        Args:
            message: Voice message to process

        Returns:
            Response text
        """
        technician_id = message.technician_id
        text = message.payload["text"]

        # Step 1: Extract intent using OpenAI
        self.logger.info("Extracting intent with OpenAI...")
        intent = await self.openai_service.extract_intent(text)

        # Step 2: Execute action based on intent
        self.logger.info(f"Executing action for intent: {intent.intent.value}")
        action_result = await self._execute_action(intent, technician_id)

        # Step 3: Get user context for RL
        user_context = self._get_or_create_user_context(technician_id)

        # Step 4: RL selects response style
        response_style = self.rl_bandit.select_arm(user_context)
        self.logger.info(f"RL selected response style: {response_style.value}")

        # Step 5: Generate response with selected style
        response_text = await self.openai_service.generate_response(
            intent=intent,
            context=action_result,
            style=response_style
        )

        return response_text

    async def _execute_action(self, intent, technician_id: str) -> dict:
        """
        Execute domain action based on extracted intent

        Args:
            intent: Extracted intent from voice input
            technician_id: ID of technician

        Returns:
            Dictionary with action result
        """
        result = {"success": False, "message": ""}

        try:
            if intent.intent == IntentType.CREATE_JOB:
                # Create new job
                job = Job(
                    customer_name=intent.customer or "Unknown Customer",
                    assigned_technician=technician_id,
                    description=intent.notes,
                    parts_used=intent.parts if intent.parts else []
                )
                job = self.job_repo.create(job)

                result = {
                    "success": True,
                    "job_id": job.job_id,
                    "customer": job.customer_name,
                    "message": f"Created job {job.job_id} for {job.customer_name}"
                }
                self.logger.info(f"Created job: {job.job_id}")

            elif intent.intent == IntentType.CLOSE_JOB:
                # Find and close job
                job = self._find_job(intent.customer, technician_id)
                if job:
                    job.status = JobStatus.COMPLETED
                    if intent.parts:
                        job.parts_used.extend(intent.parts)
                    if intent.billing_hours:
                        job.billing_hours = intent.billing_hours
                    if intent.notes:
                        job.notes.append(intent.notes)

                    job = self.job_repo.update(job)

                    result = {
                        "success": True,
                        "job_id": job.job_id,
                        "customer": job.customer_name,
                        "billing_hours": job.billing_hours,
                        "parts_used": job.parts_used,
                        "message": f"Closed job {job.job_id}, billed {job.billing_hours} hours"
                    }
                    self.logger.info(f"Closed job: {job.job_id}")
                else:
                    raise JobNotFoundError(
                        f"No active job found for customer: {intent.customer}",
                        details={"technician_id": technician_id}
                    )

            elif intent.intent == IntentType.UPDATE_JOB:
                # Update existing job
                job = self._find_job(intent.customer, technician_id)
                if job:
                    if intent.parts:
                        job.parts_used.extend(intent.parts)
                    if intent.notes:
                        job.notes.append(intent.notes)
                    if intent.billing_hours:
                        job.billing_hours += intent.billing_hours

                    job = self.job_repo.update(job)

                    result = {
                        "success": True,
                        "job_id": job.job_id,
                        "customer": job.customer_name,
                        "message": f"Updated job {job.job_id}"
                    }
                    self.logger.info(f"Updated job: {job.job_id}")
                else:
                    raise JobNotFoundError(
                        f"No active job found for customer: {intent.customer}",
                        details={"technician_id": technician_id}
                    )

            elif intent.intent == IntentType.LIST_JOBS:
                # List active jobs
                jobs = self.job_repo.list_by_technician(technician_id)
                active_jobs = [j for j in jobs if j.status != JobStatus.COMPLETED]

                result = {
                    "success": True,
                    "job_count": len(active_jobs),
                    "jobs": [
                        {
                            "id": j.job_id,
                            "customer": j.customer_name,
                            "status": j.status.value
                        }
                        for j in active_jobs[:5]  # Limit to 5 jobs
                    ],
                    "message": f"You have {len(active_jobs)} active job(s)"
                }
                self.logger.info(f"Listed {len(active_jobs)} active jobs")

            elif intent.intent == IntentType.QUERY_JOB:
                # Query specific job
                job = self._find_job(intent.customer, technician_id)
                if job:
                    result = {
                        "success": True,
                        "job_id": job.job_id,
                        "customer": job.customer_name,
                        "status": job.status.value,
                        "parts_used": job.parts_used,
                        "billing_hours": job.billing_hours,
                        "message": f"Job {job.job_id} status: {job.status.value}"
                    }
                    self.logger.info(f"Queried job: {job.job_id}")
                else:
                    result = {
                        "success": False,
                        "message": f"No job found for {intent.customer}"
                    }

            elif intent.intent == IntentType.ADD_NOTES:
                # Add notes to job
                job = self._find_job(intent.customer, technician_id)
                if job and intent.notes:
                    job.notes.append(intent.notes)
                    job = self.job_repo.update(job)

                    result = {
                        "success": True,
                        "job_id": job.job_id,
                        "message": f"Added notes to job {job.job_id}"
                    }
                    self.logger.info(f"Added notes to job: {job.job_id}")
                else:
                    result = {
                        "success": False,
                        "message": "Could not add notes"
                    }

            else:
                # Unknown intent
                result = {
                    "success": False,
                    "message": "I didn't understand that command"
                }
                self.logger.warning(f"Unknown intent: {intent.intent}")

        except JobNotFoundError as e:
            self.logger.error(f"Job not found: {e}")
            result = {
                "success": False,
                "error": "job_not_found",
                "message": str(e.message)
            }

        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            result = {
                "success": False,
                "error": type(e).__name__,
                "message": f"Failed to execute action: {str(e)}"
            }

        return result

    def _find_job(self, customer: str | None, technician_id: str) -> Job | None:
        """
        Find most recent job by customer name and technician

        Args:
            customer: Customer name to search for
            technician_id: Technician ID

        Returns:
            Most recent matching job, or None
        """
        if not customer:
            return None

        jobs = self.job_repo.list_by_technician(technician_id)

        # Find most recent job for this customer (case-insensitive)
        customer_lower = customer.lower()
        matching_jobs = [
            job for job in jobs
            if customer_lower in job.customer_name.lower() and job.status != JobStatus.COMPLETED
        ]

        if matching_jobs:
            # Sort by most recent update
            matching_jobs.sort(key=lambda j: j.updated_at, reverse=True)
            return matching_jobs[0]

        return None

    def _get_or_create_user_context(self, technician_id: str) -> UserContext:
        """
        Get or create user context for RL

        Args:
            technician_id: Technician ID

        Returns:
            User context
        """
        if technician_id not in self.user_contexts:
            self.user_contexts[technician_id] = UserContext(
                technician_id=technician_id,
                time_of_day=datetime.now(timezone.utc).hour
            )

        # Update time of day
        context = self.user_contexts[technician_id]
        context.time_of_day = datetime.now(timezone.utc).hour

        return context

    def _update_user_context(
        self,
        technician_id: str,
        response_time: float,
        error_occurred: bool,
        user_repeated: bool
    ) -> None:
        """
        Update user context after interaction and calculate RL reward

        Args:
            technician_id: Technician ID
            response_time: Time taken to respond
            error_occurred: Whether an error occurred
            user_repeated: Whether user had to repeat
        """
        context = self._get_or_create_user_context(technician_id)

        # Update interaction count
        context.interaction_count += 1

        # Update error count
        if error_occurred:
            context.recent_errors += 1
        else:
            # Decay error count over successful interactions
            context.recent_errors = max(0, context.recent_errors - 1)

        # Update average response time (exponential moving average)
        alpha = 0.3
        context.avg_response_time = (
            alpha * response_time + (1 - alpha) * context.avg_response_time
        )

        # Calculate reward for RL
        reward = self.rl_bandit.calculate_implicit_reward(
            response_time=response_time,
            error_occurred=error_occurred,
            user_repeated=user_repeated
        )

        # Update bandit with reward (need to know which arm was used)
        # Note: In a real implementation, we'd track the arm used in this interaction
        # For now, we'll skip the reward update here and handle it separately

        self.user_contexts[technician_id] = context

        self.logger.debug(
            f"Updated context for {technician_id}: "
            f"interactions={context.interaction_count}, "
            f"errors={context.recent_errors}, "
            f"avg_time={context.avg_response_time:.2f}s"
        )

    async def start_worker_mode(self) -> None:
        """Start as background worker consuming from Service Bus"""
        self.logger.info("Starting worker mode...")

        consumer = AzureServiceBusConsumer(
            settings=self.settings,
            message_handler=self._process_message
        )

        try:
            await consumer.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Worker mode interrupted by user")
            consumer.stop()
        except Exception as e:
            self.logger.error(f"Worker mode error: {e}")
            raise

    def get_statistics(self) -> dict:
        """Get system statistics"""
        return {
            "total_jobs": self.job_repo.count(),
            "active_users": len(self.user_contexts),
            "rl_stats": self.rl_bandit.get_statistics()
        }

    def __repr__(self) -> str:
        return f"DispatchAgent(jobs={self.job_repo.count()}, users={len(self.user_contexts)})"
