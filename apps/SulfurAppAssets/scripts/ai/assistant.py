"""
YouTube Creator Assistant - Gemini AI Helper
A modular script for generating AI responses to YouTube creator questions.

Usage:
    from youtube_creator_assistant import generate_reply

    response = generate_reply(
        user_message="How can I improve my thumbnails?",
        context={"video_title": "My Latest Video", "views": 1000},
        api_key="your_gemini_api_key"
    )
"""

from google import genai
from typing import Dict, Any, Optional
import json


class YouTubeCreatorAssistant:
    """AI assistant for YouTube creators using Gemini."""

    def __init__(self, api_key: str):
        """
        Initialize the assistant with a Gemini API key.

        Args:
            api_key: Gemini API key from https://aistudio.google.com/app/apikey
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate_reply(
            self,
            user_message: str,
            context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Generate an AI reply to a user's message with optional context.

        Args:
            user_message: The user's question or message
            context: Optional context data to include (video stats, channel info, etc.)
            temperature: Controls randomness (0.0-1.0, default 0.7)
            max_tokens: Maximum response length (default 2048)

        Returns:
            Dict with 'response', 'success', and optional 'error' keys
        """
        try:
            # Build the prompt with context
            prompt = self._build_prompt(user_message, context)

            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            return {
                "response": response.text,
                "success": True,
                "context_included": context is not None
            }

        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }

    def _build_prompt(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a comprehensive prompt with system instructions and context.

        Args:
            user_message: The user's question
            context: Optional context data

        Returns:
            Formatted prompt string
        """
        # Base system prompt for YouTube creators
        system_prompt = """You are an expert YouTube growth consultant and content strategist. Your role is to provide actionable, data-driven advice to YouTube creators.

Key principles:
- Be specific and actionable - avoid generic advice
- Reference data and metrics when available
- Consider the YouTube algorithm and best practices
- Provide 2-3 concrete next steps when relevant
- Be encouraging but honest about challenges
- Focus on sustainable growth strategies

Your responses should be:
- Concise (1-3 paragraphs unless more detail is requested)
- Data-informed (use any provided context/stats)
- Practical (focus on what creators can actually implement)
- Balanced (acknowledge both strengths and areas for improvement)
"""

        # Add context if provided
        context_section = ""
        if context:
            context_section = "\n\nCONTEXT INFORMATION:\n"
            context_section += self._format_context(context)

        # Combine into final prompt
        full_prompt = f"""{system_prompt}{context_section}

USER QUESTION:
{user_message}

Provide a helpful, actionable response based on the question and any context provided above."""

        return full_prompt

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context data into a readable string for the prompt.

        Args:
            context: Dictionary of context data

        Returns:
            Formatted context string
        """
        formatted = []

        # Video-specific context
        if "video" in context:
            video = context["video"]
            formatted.append(f"Video Title: {video.get('title', 'N/A')}")
            if "video_id" in video:
                formatted.append(f"Video ID: {video['video_id']}")

        # Stats context
        if "stats" in context:
            stats = context["stats"]
            formatted.append("\nVideo Statistics:")
            if "view_count" in stats:
                formatted.append(f"  - Views: {stats['view_count']:,}")
            if "like_count" in stats:
                formatted.append(f"  - Likes: {stats['like_count']:,}")
            if "comment_count" in stats:
                formatted.append(f"  - Comments: {stats['comment_count']:,}")
            if "video_age_days" in stats:
                formatted.append(f"  - Age: {stats['video_age_days']} days")
            if "views_per_day" in stats:
                formatted.append(f"  - Views/day: {stats['views_per_day']:,.1f}")
            if "like_rate" in stats:
                formatted.append(f"  - Like rate: {stats['like_rate']:.4f}")
            if "comment_rate" in stats:
                formatted.append(f"  - Comment rate: {stats['comment_rate']:.4f}")

        # Growth prediction context (from predictor_template stages)
        if "stage2" in context:
            stage2 = context["stage2"]
            formatted.append("\nGrowth Analysis:")
            if "composite_growth_score" in stage2:
                formatted.append(f"  - Growth Score: {stage2['composite_growth_score']:.3f}")
            if "components" in stage2:
                comps = stage2["components"]
                formatted.append(f"  - Engagement: {comps.get('engagement_score', 0):.2f}")
                formatted.append(f"  - Virality: {comps.get('virality_score', 0):.2f}")
                formatted.append(f"  - Authority: {comps.get('authority_score', 0):.2f}")

        if "stage3" in context:
            stage3 = context["stage3"]
            formatted.append("\nGrowth Predictions:")
            if "predicted_30d" in stage3:
                formatted.append(f"  - Predicted 30-day views: {stage3['predicted_30d']:,}")
            if "baseline" in stage3 and "ceiling" in stage3:
                formatted.append(f"  - Range: {stage3['baseline']:,} - {stage3['ceiling']:,}")
            if "decay_pattern" in stage3:
                formatted.append(f"  - Growth pattern: {stage3['decay_pattern']}")
            if "momentum_factor" in stage3:
                formatted.append(f"  - Momentum: {stage3['momentum_factor']:.3f}")

        # Channel context
        if "channel_subscribers" in context:
            formatted.append(f"\nChannel Subscribers: {context['channel_subscribers']:,}")
        if "channel_average_views" in context:
            formatted.append(f"Channel Average Views: {context['channel_average_views']:,}")

        # Custom context (any other keys)
        custom_keys = set(context.keys()) - {"video", "stats", "stage2", "stage3", "channel_subscribers",
                                             "channel_average_views"}
        if custom_keys:
            formatted.append("\nAdditional Context:")
            for key in custom_keys:
                value = context[key]
                if isinstance(value, (dict, list)):
                    formatted.append(f"  - {key}: {json.dumps(value)}")
                else:
                    formatted.append(f"  - {key}: {value}")

        return "\n".join(formatted)


# Convenience function for simple usage
def generate_reply(
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Generate an AI reply to a YouTube creator's question.

    This is a convenience function that creates a temporary assistant instance.
    For multiple calls, create a YouTubeCreatorAssistant instance instead.

    Args:
        user_message: The user's question or message
        context: Optional context data (video stats, predictions, etc.)
        api_key: Gemini API key (required)
        temperature: Controls randomness (0.0-1.0, default 0.7)
        max_tokens: Maximum response length (default 2048)

    Returns:
        Dict with 'response', 'success', and optional 'error' keys

    Example:
        >>> context = {
        ...     "video": {"title": "My Video", "video_id": "abc123"},
        ...     "stats": {"view_count": 5000, "video_age_days": 7}
        ... }
        >>> result = generate_reply("How is my video performing?", context, api_key)
        >>> print(result["response"])
    """
    if not api_key:
        return {
            "response": "Error: API key is required. Get one from https://aistudio.google.com/app/apikey",
            "success": False,
            "error": "No API key provided"
        }

    assistant = YouTubeCreatorAssistant(api_key)
    return assistant.generate_reply(user_message, context, temperature, max_tokens)


# Example usage and testing
if __name__ == "__main__":
    import os

    # Get API key from environment or prompt
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()

    if not api_key:
        print("Error: API key required")
        exit(1)

    # Example 1: Simple question without context
    print("\n" + "=" * 80)
    print("Example 1: Simple question")
    print("=" * 80)

    result = generate_reply(
        "What are the best practices for YouTube thumbnails?",
        api_key=api_key
    )

    if result["success"]:
        print(result["response"])
    else:
        print(f"Error: {result['error']}")

    # Example 2: Question with video context
    print("\n" + "=" * 80)
    print("Example 2: Question with video context")
    print("=" * 80)

    context = {
        "video": {
            "title": "How to Make the Perfect Pizza",
            "video_id": "example123"
        },
        "stats": {
            "view_count": 15000,
            "like_count": 450,
            "comment_count": 85,
            "video_age_days": 14,
            "views_per_day": 1071.4,
            "like_rate": 0.03,
            "comment_rate": 0.0057
        },
        "stage2": {
            "composite_growth_score": 0.685,
            "components": {
                "engagement_score": 0.72,
                "virality_score": 0.65,
                "authority_score": 0.58
            }
        },
        "stage3": {
            "predicted_30d": 45000,
            "baseline": 35000,
            "ceiling": 60000,
            "decay_pattern": "steady",
            "momentum_factor": 0.85
        }
    }

    result = generate_reply(
        "How is this video performing and what should I do next?",
        context=context,
        api_key=api_key
    )

    if result["success"]:
        print(result["response"])
    else:
        print(f"Error: {result['error']}")

    # Example 3: Using the assistant class for multiple calls
    print("\n" + "=" * 80)
    print("Example 3: Multiple questions with persistent assistant")
    print("=" * 80)

    assistant = YouTubeCreatorAssistant(api_key)

    questions = [
        "What's the ideal video length for gaming content?",
        "How often should I upload to maintain growth?",
        "Should I focus on shorts or long-form content?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        result = assistant.generate_reply(question)
        if result["success"]:
            print(f"A{i}: {result['response']}\n")
        else:
            print(f"Error: {result['error']}\n")