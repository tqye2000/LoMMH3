import json
import types

import openai
import pytest

from ui.core import prompt_ai
from ui.core.settings import load_settings


def test_revise_prompt_uses_responses_api(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, api_key, timeout):
            calls["api_key"] = api_key
            self.responses = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(output_text="  A revised scene.  ")

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    text = prompt_ai.revise_prompt("A cat.", " sk-test ", "gpt-test", "more rain")
    assert text == "A revised scene."
    assert calls["api_key"] == "sk-test" and calls["model"] == "gpt-test"
    assert "more rain" in calls["input"] and "A cat." in calls["input"]


def test_revise_prompt_includes_previous_scenes(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, api_key, timeout):
            self.responses = types.SimpleNamespace(create=lambda **kw: calls.update(kw) or
                                                   types.SimpleNamespace(output_text="ok"))

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    prompt_ai.revise_prompt("Scene three.", "sk-test", previous=[("Intro", "A red fox."), ("Chase", "It runs.")])
    text = calls["input"]
    assert text.index("A red fox.") < text.index("It runs.") < text.index("Scene three.")
    assert "Intro" in text and "Chase" in text


def test_previous_prompts_in_step_order(service):
    pid = service.create_project("Story").id
    first, empty, third, fourth = (service.add_step(pid, n).id for n in ("One", "Two", "Three", "Four"))
    service.save_prompt(pid, first, "First.")
    service.save_prompt(pid, third, "Third.")
    service.save_prompt(pid, fourth, "Fourth.")
    assert service.previous_prompts(pid, first) == []
    assert service.previous_prompts(pid, fourth) == [("One", "First."), ("Three", "Third.")]


@pytest.mark.parametrize("prompt,key", [("A cat.", ""), ("  ", "sk-test")])
def test_revise_prompt_validates_inputs(prompt, key):
    with pytest.raises(prompt_ai.PromptAIError):
        prompt_ai.revise_prompt(prompt, key)


def test_openai_settings_from_file_and_env(monkeypatch, tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps({"openai_api_key": "sk-file", "openai_model": "gpt-file"}), encoding="utf-8")
    monkeypatch.setenv("MINIMAX_UI_DATA", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_UI_OPENAI_MODEL", raising=False)
    settings = load_settings()
    assert settings.openai_api_key.get_secret_value() == "sk-file" and settings.openai_model == "gpt-file"
    assert "sk-file" not in repr(settings)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert load_settings().openai_api_key.get_secret_value() == "sk-env"
