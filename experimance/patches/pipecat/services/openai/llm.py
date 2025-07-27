#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM service implementation with context aggregators."""

import json
from dataclasses import dataclass
from typing import Any, Optional

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.base_llm import BaseOpenAILLMService


@dataclass
class OpenAIContextAggregatorPair:
    """Pair of OpenAI context aggregators for user and assistant messages.

    Parameters:
        _user: User context aggregator for processing user messages.
        _assistant: Assistant context aggregator for processing assistant messages.
    """

    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    """OpenAI LLM service implementation with enhanced shutdown handling.

    Provides a complete OpenAI LLM service with context aggregation support.
    Uses the BaseOpenAILLMService for core functionality and adds OpenAI-specific
    context aggregator creation, plus enhanced cleanup to prevent hanging HTTP connections.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        params: Optional[BaseOpenAILLMService.InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI LLM service.

        Args:
            model: The OpenAI model name to use. Defaults to "gpt-4.1".
            params: Input parameters for model configuration.
            **kwargs: Additional arguments passed to the parent BaseOpenAILLMService.
        """
        super().__init__(model=model, params=params, **kwargs)
        self._shutdown_state = "running"

    async def stop(self, frame):
        """Stop the OpenAI LLM service and clean up HTTP connections.
        
        Args:
            frame: The EndFrame that triggered the stop.
        """
        logger.debug("🔧 PATCH: OpenAI LLM service stop() called")
        await super().stop(frame)
        await self._disconnect(force=False)

    async def _disconnect(self, force: bool = False):
        """Disconnect and clean up HTTP client connections.
        
        Args:
            force: If True, forces immediate disconnection without waiting for graceful close.
        """
        if self._shutdown_state != "running":
            logger.debug("OpenAI LLM service already disconnecting/disconnected")
            return
            
        self._shutdown_state = "disconnecting"
        logger.debug("🔧 PATCH: OpenAI LLM service disconnect started (enhanced shutdown)")
        
        try:
            # Close the HTTP client if it exists
            if hasattr(self, '_client') and self._client:
                # Access the underlying HTTP client
                http_client = getattr(self._client, 'http_client', None)
                if http_client:
                    if force:
                        # Force close without waiting
                        logger.debug("🔧 PATCH: Force closing OpenAI HTTP client")
                        try:
                            # Try to close immediately
                            if hasattr(http_client, 'aclose'):
                                await asyncio.wait_for(http_client.aclose(), timeout=0.1)
                            else:
                                logger.debug("🔧 PATCH: HTTP client has no aclose method")
                        except asyncio.TimeoutError:
                            logger.debug("🔧 PATCH: HTTP client close timeout, proceeding")
                        except Exception as e:
                            logger.debug(f"🔧 PATCH: HTTP client force close error: {e}")
                    else:
                        # Graceful close with timeout
                        logger.debug("🔧 PATCH: Gracefully closing OpenAI HTTP client")
                        try:
                            if hasattr(http_client, 'aclose'):
                                await asyncio.wait_for(http_client.aclose(), timeout=2.0)
                            else:
                                logger.debug("🔧 PATCH: HTTP client has no aclose method")
                        except asyncio.TimeoutError:
                            logger.debug("🔧 PATCH: HTTP client graceful close timeout, forcing")
                            # Try force close as fallback
                            try:
                                if hasattr(http_client, 'aclose'):
                                    await asyncio.wait_for(http_client.aclose(), timeout=0.1)
                            except Exception as e:
                                logger.debug(f"🔧 PATCH: HTTP client fallback close error: {e}")
                        except Exception as e:
                            logger.debug(f"🔧 PATCH: HTTP client graceful close error: {e}")
                else:
                    logger.debug("🔧 PATCH: No HTTP client found in OpenAI client")
                    
                # Clear the client reference
                self._client = None
                logger.debug("🔧 PATCH: OpenAI client reference cleared")
            else:
                logger.debug("🔧 PATCH: No OpenAI client to disconnect")
                
        except Exception as e:
            logger.error(f"🔧 PATCH: Error during OpenAI LLM disconnect: {e}")
        finally:
            self._shutdown_state = "disconnected"
            logger.debug("🔧 PATCH: OpenAI LLM service disconnect completed")

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create OpenAI-specific context aggregators.

        Creates a pair of context aggregators optimized for OpenAI's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    """OpenAI-specific user context aggregator.

    Handles aggregation of user messages for OpenAI LLM services.
    Inherits all functionality from the base LLMUserContextAggregator.
    """

    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    """OpenAI-specific assistant context aggregator.

    Handles aggregation of assistant messages for OpenAI LLM services,
    with specialized support for OpenAI's function calling format,
    tool usage tracking, and image message handling.
    """

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call in progress.

        Adds the function call to the context with an IN_PROGRESS status
        to track ongoing function execution.

        Args:
            frame: Frame containing function call progress information.
        """
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle the result of a function call.

        Updates the context with the function call result, replacing any
        previous IN_PROGRESS status.

        Args:
            frame: Frame containing the function call result.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle a cancelled function call.

        Updates the context to mark the function call as cancelled.

        Args:
            frame: Frame containing the function call cancellation information.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if (
                message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle a user image frame from a function call request.

        Marks the associated function call as completed and adds the image
        to the context for processing.

        Args:
            frame: Frame containing the user image and request context.
        """
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )
