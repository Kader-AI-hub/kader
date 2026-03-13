"""
Session title generation utilities.

Provides functions to generate brief, descriptive titles for conversation
sessions to help users find them later.
"""

from loguru import logger

from kader.prompts.agent_prompts import SessionTitlePrompt
from kader.providers.base import BaseLLMProvider, Message, ModelConfig


def generate_session_title(
    provider: BaseLLMProvider,
    query: str,
) -> str:
    """
    Generate a brief title for a conversation session.

    Args:
        provider: LLM provider to use for title generation
        query: The query as a string

    Returns:
        A brief descriptive title for the conversation
    """

    prompt = SessionTitlePrompt(query=query)
    system_prompt = prompt.resolve_prompt()

    config = ModelConfig(temperature=0.3, max_tokens=1000)

    messages = [Message.system(content=system_prompt)]

    try:
        response = provider.invoke(messages, config)
        title = response.content.strip()
        logger.debug(f"Generated session title: {title}")
        return title
    except Exception as e:
        logger.error(f"Failed to generate session title: {e}")
        return "Untitled Session"


async def agenerate_session_title(
    provider: BaseLLMProvider,
    query: str,
) -> str:
    """
    Generate a brief title for a conversation session asynchronously.

    Args:
        provider: LLM provider to use for title generation
        query: The query as a string

    Returns:
        A brief descriptive title for the conversation
    """

    prompt = SessionTitlePrompt(query=query)
    system_prompt = prompt.resolve_prompt()

    config = ModelConfig(temperature=0.3, max_tokens=1000)

    messages = [Message.system(content=system_prompt)]

    try:
        response = await provider.ainvoke(messages, config)
        title = response.content.strip()
        logger.debug(f"Generated session title: {title}")
        return title
    except Exception as e:
        logger.error(f"Failed to generate session title: {e}")
        return "Untitled Session"
