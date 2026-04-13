from __future__ import annotations

REASONING_MODELS: frozenset[str] = frozenset(
    {
        # GPT 5.4 Family
        "gpt-5.4",
        "gpt-5.4-pro",
        # GPT 5.3 Family
        "gpt-5.3-codex",
        # GPT 5.2 Family
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.2-codex",
        # GPT 5.1 Family
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        # GPT 5 Family
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-codex",
        # o Family
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }
)

CODING_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5-codex",
    }
)
