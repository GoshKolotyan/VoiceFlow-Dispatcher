import asyncio
import json
from typing import Callable, Awaitable
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

from src.core.logger import LoggerFactory
from src.core.model import VoiceMessage
from src.core.expections import ServiceBusError
from src.config.settings import Settings


class AzureServiceBusProducer:
    """Async producer for Azure Service Bus"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = LoggerFactory.create_logger("ServiceBusProducer", level=settings.LOG_LEVEL)
        self.queue_name = settings.SERVICEBUS_QUEUE_NAME

        # Create async client
        self.client = ServiceBusClient.from_connection_string(
            conn_str=settings.AZURE_SERVICEBUS_CONN_STR,
            logging_enable=True
        )

        self.logger.info(f"Initialized Service Bus producer for queue: {self.queue_name}")

    async def send_message(self, message: VoiceMessage) -> None:
        """Send a single message to the queue"""
        try:
            async with self.client:
                sender = self.client.get_queue_sender(queue_name=self.queue_name)
                async with sender:
                    # Serialize message to JSON
                    message_json = message.model_dump_json()
                    sb_message = ServiceBusMessage(
                        body=message_json,
                        content_type="application/json",
                        message_id=message.message_id
                    )

                    await sender.send_messages(sb_message)
                    self.logger.info(
                        f"Sent message {message.message_id} to queue {self.queue_name}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise ServiceBusError(
                "Failed to send message to Service Bus",
                details={
                    "message_id": message.message_id,
                    "queue": self.queue_name,
                    "error": str(e)
                }
            )

    async def send_batch(self, messages: list[VoiceMessage]) -> None:
        """Send batch of messages for higher throughput"""
        if not messages:
            self.logger.warning("Attempted to send empty batch")
            return

        try:
            async with self.client:
                sender = self.client.get_queue_sender(queue_name=self.queue_name)
                async with sender:
                    batch = await sender.create_message_batch()

                    for msg in messages:
                        message_json = msg.model_dump_json()
                        sb_message = ServiceBusMessage(
                            body=message_json,
                            content_type="application/json",
                            message_id=msg.message_id
                        )

                        try:
                            batch.add_message(sb_message)
                        except ValueError:
                            # Batch is full, send it and create a new batch
                            await sender.send_messages(batch)
                            batch = await sender.create_message_batch()
                            batch.add_message(sb_message)

                    # Send remaining messages in batch
                    if len(batch) > 0:
                        await sender.send_messages(batch)

                    self.logger.info(
                        f"Sent batch of {len(messages)} messages to queue {self.queue_name}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to send batch: {e}")
            raise ServiceBusError(
                "Failed to send message batch",
                details={
                    "message_count": len(messages),
                    "queue": self.queue_name,
                    "error": str(e)
                }
            )

    async def close(self) -> None:
        """Close the Service Bus client"""
        await self.client.close()
        self.logger.info("Service Bus producer closed")

    def __repr__(self) -> str:
        return f"AzureServiceBusProducer(queue={self.queue_name})"


class AzureServiceBusConsumer:
    """Async consumer for Azure Service Bus"""

    def __init__(
        self,
        settings: Settings,
        message_handler: Callable[[VoiceMessage], Awaitable[None]]
    ):
        self.settings = settings
        self.logger = LoggerFactory.create_logger("ServiceBusConsumer", level=settings.LOG_LEVEL)
        self.queue_name = settings.SERVICEBUS_QUEUE_NAME
        self.message_handler = message_handler
        self.is_running = False

        # Create async client
        self.client = ServiceBusClient.from_connection_string(
            conn_str=settings.AZURE_SERVICEBUS_CONN_STR,
            logging_enable=True
        )

        self.logger.info(f"Initialized Service Bus consumer for queue: {self.queue_name}")

    async def start_consuming(self) -> None:
        """Start consuming messages from queue in a long-running loop"""
        self.is_running = True
        self.logger.info("Starting message consumer...")

        try:
            async with self.client:
                receiver = self.client.get_queue_receiver(
                    queue_name=self.queue_name,
                    max_wait_time=5
                )

                async with receiver:
                    while self.is_running:
                        try:
                            # Receive batch of messages
                            messages = await receiver.receive_messages(
                                max_message_count=10,
                                max_wait_time=5
                            )

                            if not messages:
                                # No messages, short sleep to prevent tight loop
                                await asyncio.sleep(0.1)
                                continue

                            # Process each message
                            for msg in messages:
                                try:
                                    # Deserialize message
                                    message_body = str(msg)
                                    message_data = json.loads(message_body)
                                    voice_message = VoiceMessage(**message_data)

                                    self.logger.info(
                                        f"Processing message {voice_message.message_id}"
                                    )

                                    # Call the handler
                                    await self.message_handler(voice_message)

                                    # Complete (delete) message on success
                                    await receiver.complete_message(msg)
                                    self.logger.info(
                                        f"Completed message {voice_message.message_id}"
                                    )

                                except json.JSONDecodeError as e:
                                    self.logger.error(f"Failed to decode message JSON: {e}")
                                    # Dead-letter invalid messages
                                    await receiver.dead_letter_message(
                                        msg,
                                        reason="InvalidJSON",
                                        error_description=str(e)
                                    )

                                except Exception as e:
                                    self.logger.error(f"Failed to process message: {e}")
                                    # Dead-letter messages that fail processing
                                    await receiver.dead_letter_message(
                                        msg,
                                        reason="ProcessingError",
                                        error_description=str(e)
                                    )

                        except Exception as e:
                            self.logger.error(f"Error receiving messages: {e}")
                            # Wait before retry
                            await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Consumer error: {e}")
            raise ServiceBusError(
                "Consumer failed",
                details={"queue": self.queue_name, "error": str(e)}
            )
        finally:
            self.logger.info("Consumer stopped")

    def stop(self) -> None:
        """Signal the consumer to stop processing"""
        self.is_running = False
        self.logger.info("Stopping consumer...")

    async def close(self) -> None:
        """Close the Service Bus client"""
        self.stop()
        await self.client.close()
        self.logger.info("Service Bus consumer closed")

    def __repr__(self) -> str:
        return f"AzureServiceBusConsumer(queue={self.queue_name})"
