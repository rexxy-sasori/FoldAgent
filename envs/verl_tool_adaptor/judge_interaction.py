import json
import logging
import os
import re
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from envs.local_search import judge, parse_judge_response

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_question(messages: list[dict[str, Any]]) -> Optional[str]:
    """Extract the original question from conversation messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        The question/problem statement or None if not found
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content and not content.startswith("[") and not content.startswith("docid:"):
                return content
    return None


def extract_answer(messages: list[dict[str, Any]]) -> Optional[str]:
    """Extract the predicted answer from conversation messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        The last assistant message or None if not found
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and not content.startswith("[") and not content.startswith("docid:"):
                return content
    return None


def extract_label_answer(extra_kwargs: dict[str, Any]) -> Optional[str]:
    """Extract the label/ground truth answer from extra_kwargs.
    
    Args:
        extra_kwargs: Extra keyword arguments containing label
        
    Returns:
        The label answer or None if not found
    """
    if "label_answer" in extra_kwargs:
        return extra_kwargs["label_answer"]
    if "problem_statement" in extra_kwargs and "answer" in extra_kwargs:
        return extra_kwargs.get("answer")
    return None


def extract_problem_statement(extra_kwargs: dict[str, Any]) -> Optional[str]:
    """Extract the problem statement from extra_kwargs.
    
    Args:
        extra_kwargs: Extra keyword arguments containing problem
        
    Returns:
        The problem statement or None if not found
    """
    if "problem_statement" in extra_kwargs:
        return extra_kwargs["problem_statement"]
    return None


class JudgeInteraction(BaseInteraction):
    """Interaction that evaluates agent responses using judge models.
    
    This interaction uses the existing judge system from envs.local_search
    to evaluate agent responses against ground truth labels.
    
    The judge can be configured via environment variable JUDGE_OPENAI_MODEL.
    """

    def __init__(self, config: dict):
        """Initialize JudgeInteraction.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.judge_model = os.getenv("JUDGE_OPENAI_MODEL", "gpt-4.1")
        logger.info(f"Initialized JudgeInteraction with judge_model: {self.judge_model}")

    async def start_interaction(self, request_id: str, **kwargs):
        """Start interaction session.
        
        Args:
            request_id: Unique identifier for this interaction
            **kwargs: Additional interaction parameters
        """
        logger.debug(f"[JudgeInteraction] Started interaction for request_id={request_id}")
        pass

    async def generate_response(
        self,
        request_id: str,
        messages: list[dict[str, Any]],
        **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Generate evaluation by judging agent response.
        
        Args:
            request_id: Unique identifier for this interaction
            messages: Conversation history
            **kwargs: Additional parameters including label_answer and problem_statement
            
        Returns:
            Tuple of (should_terminate, response, reward, metrics)
                - should_terminate: Whether to terminate the sequence
                - response: Empty string (no user response needed)
                - reward: Judge score (0.0 to 1.0)
                - metrics: Additional metrics including judge model and details
        """
        try:
            question = extract_problem_statement(kwargs) or extract_question(messages)
            predicted_answer = extract_answer(messages)
            label_answer = extract_label_answer(kwargs)
            
            if not question:
                logger.warning(f"[JudgeInteraction] No question found in messages or kwargs")
                return True, "", 0.0, {"error": "no_question"}
            
            if not predicted_answer:
                logger.warning(f"[JudgeInteraction] No predicted answer found in messages")
                return True, "", 0.0, {"error": "no_predicted_answer"}
            
            if not label_answer:
                logger.warning(f"[JudgeInteraction] No label answer found in kwargs")
                return True, "", 0.0, {"error": "no_label_answer"}
            
            logger.info(f"[JudgeInteraction] Judging response for request_id={request_id}")
            logger.debug(f"  Question: {question[:100]}...")
            logger.debug(f"  Predicted: {predicted_answer[:100]}...")
            logger.debug(f"  Label: {label_answer[:100]}...")
            
            reward = await judge(question, label_answer, predicted_answer, model=self.judge_model)
            
            metrics = {
                "judge_model": self.judge_model,
                "question_length": len(question),
                "predicted_length": len(predicted_answer),
                "label_length": len(label_answer),
                "reward": reward,
            }
            
            logger.info(f"[JudgeInteraction] Reward: {reward} for request_id={request_id}")
            
            return True, "", reward, metrics
            
        except Exception as e:
            logger.error(f"[JudgeInteraction] Error during judgment: {e}")
            return True, "", 0.0, {"error": str(e), "judge_model": self.judge_model}

    async def end_interaction(self, request_id: str, **kwargs):
        """End interaction session.
        
        Args:
            request_id: Unique identifier for this interaction
            **kwargs: Additional interaction parameters
        """
        logger.debug(f"[JudgeInteraction] Ended interaction for request_id={request_id}")
        pass
