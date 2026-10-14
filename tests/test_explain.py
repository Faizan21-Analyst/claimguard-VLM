import json
from types import SimpleNamespace

from claimguard.agent.explain import summarize
from claimguard.agent.fusion import FusionResult
from claimguard.schemas.claim import Claim


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=response["finish_reason"],
                    message=SimpleNamespace(content=response["content"]),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_inputs():
    claim = Claim(
        claim_id="claim_test",
        policy_id="policy_A",
        claimed_damage_type="hail",
        component="roof_shingles",
        incident_date="2026-06-10",
    )
    result = FusionResult(
        decision="supported",
        reasons=["Photos show hail damage.", "Clause 4.2 covers hail damage."],
    )
    return result, claim


def reply(content, finish_reason="stop"):
    return {"content": content, "finish_reason": finish_reason}


def test_summarizer_returns_normal_reply():
    result, claim = make_inputs()
    client = FakeClient([reply("The photos show hail damage. Policy evidence supports review.")])

    summary = summarize(result, claim, client=client)

    assert summary.startswith("The photos show hail damage.")
    assert len(client.chat.completions.calls) == 1


def test_summarizer_strips_think_block():
    result, claim = make_inputs()
    client = FakeClient([reply("<think>Internal reasoning</think>\nThe claim has supporting evidence.")])

    summary = summarize(result, claim, client=client)

    assert "<think>" not in summary
    assert "Internal reasoning" not in summary
    assert "supporting evidence" in summary


def test_summarizer_returns_empty_after_api_errors():
    result, claim = make_inputs()
    client = FakeClient([RuntimeError("Primary API failed"), RuntimeError("Fallback API failed")])

    assert summarize(result, claim, client=client) == ""
    assert len(client.chat.completions.calls) == 2


def test_summarizer_rejects_truncated_reply():
    result, claim = make_inputs()
    client = FakeClient([
        reply("Incomplete summary", finish_reason="length"),
        reply("", finish_reason="stop"),
    ])

    assert summarize(result, claim, client=client) == ""
    assert len(client.chat.completions.calls) == 2


def test_fallback_model_is_used_when_primary_fails():
    result, claim = make_inputs()
    client = FakeClient([RuntimeError("primary down"), reply("Fallback summary.")])

    assert summarize(result, claim, client=client) == "Fallback summary."

    models = [call["model"] for call in client.chat.completions.calls]
    assert models[0] != models[1]


def test_unclosed_think_block_is_rejected():
    result, claim = make_inputs()
    client = FakeClient([reply("<think>still thinking..."), reply("Fallback summary.")])

    assert summarize(result, claim, client=client) == "Fallback summary."


def test_prompt_contains_decision_and_reasons():
    result, claim = make_inputs()
    client = FakeClient([reply("ok")])

    summarize(result, claim, client=client)

    prompt = json.dumps(client.chat.completions.calls[0]["messages"])
    assert "supported" in prompt
    assert "Clause 4.2 covers hail damage." in prompt