"""Revise a generation prompt with an OpenAI model."""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_MODEL = "gpt-5"

SYSTEM_INSTRUCTIONS = (
    "You revise prompts for an audio+video generation model. Improve clarity and vividness: "
    "describe subjects, action, setting, camera movement, lighting, and sound/dialogue concretely. "
    "Keep the original language, intent, and any 'Picture N', 'Video N', 'Audio N', or 'Subject N' "
    "references unchanged. "
    "If prompts of previous scenes are given, they are context only: keep characters, appearance, "
    "wardrobe, setting, style, and story continuous with them, but do not repeat or revise them. "
    "Follow the user's extra instructions if given. Return only the revised prompt text."
)


class PromptAIError(Exception):
    pass


def revise_prompt(prompt: str, api_key: str, model: str = DEFAULT_MODEL, instruction: str = "",
                  previous: Sequence[tuple[str, str]] = ()) -> str:
    if not api_key.strip():
        raise PromptAIError("No OpenAI API key configured: set openai_api_key in settings.json "
                            "or the OPENAI_API_KEY environment variable on the server.")
    if not prompt.strip():
        raise PromptAIError("The prompt is empty.")
    try:
        import openai
    except ImportError as exc:
        raise PromptAIError("The 'openai' package is not installed in the UI environment.") from exc

    user_input = f"Prompt:\n{prompt}"
    if previous:
        scenes = "\n\n".join(f"Scene {i} ({name}):\n{text.strip()}" for i, (name, text) in enumerate(previous, 1))
        user_input = f"Previous scenes (context only):\n{scenes}\n\n{user_input}"
    if instruction.strip():
        user_input = f"Instructions: {instruction.strip()}\n\n{user_input}"
    try:
        client = openai.OpenAI(api_key=api_key.strip(), timeout=120)
        response = client.responses.create(
            model=model.strip() or DEFAULT_MODEL, instructions=SYSTEM_INSTRUCTIONS, input=user_input)
    except openai.OpenAIError as exc:
        raise PromptAIError(f"OpenAI request failed: {exc}") from exc
    text = (response.output_text or "").strip()
    if not text:
        raise PromptAIError("The model returned an empty response.")
    return text
