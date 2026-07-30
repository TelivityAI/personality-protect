"""Tests for the locked RAG writing prompt (chat turns + flat fallback)."""

from personality_protect.prompt_write import (
    WRITE_SYSTEM_PROMPT,
    build_write_messages,
    build_write_prompt,
)


def test_build_write_messages_are_system_plus_user():
    messages = build_write_messages(
        topic="Contoso's quarterly planning",
        points="- Keep the plan focused\n- Name one accountable owner",
        examples=[
            "Short lines.\nClear decisions.",
            "A plan is useful only when the owner is named.",
        ],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == WRITE_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "EXAMPLES (rhythm reference only" in user
    assert "Short lines.\nClear decisions." in user
    assert "A plan is useful only when the owner is named." in user
    assert "BRIEF:\nTopic: Contoso's quarterly planning\n" in user
    assert "- Keep the plan focused" in user
    assert "Write one new post from the BRIEF now" in user
    # No trailing "POST:" completion cue — the chat template's assistant
    # turn marker is the generation prompt, and "POST:" was scaffolding
    # the old path echoed.
    assert "POST:" not in user


def test_build_write_prompt_flat_fallback_matches_locked_content():
    prompt = build_write_prompt(
        topic="Contoso's quarterly planning",
        points="- Keep the plan focused\n- Name one accountable owner",
        examples=[
            "Short lines.\nClear decisions.",
            "A plan is useful only when the owner is named.",
        ],
    )

    assert prompt.startswith(WRITE_SYSTEM_PROMPT)
    assert "EXAMPLES (rhythm reference only" in prompt
    assert "BRIEF:\nTopic: Contoso's quarterly planning\n" in prompt
    assert prompt.rstrip().endswith("Do not repeat any EXAMPLE.")


def test_bare_base_omits_examples_header_entirely():
    """An empty EXAMPLES: header is scaffolding the model will echo."""
    messages = build_write_messages(
        topic="Contoso product updates",
        points="- Explain what changed",
        examples=[],
    )
    user = messages[1]["content"]
    assert "EXAMPLES" not in user
    assert user.startswith("BRIEF:\n")
    assert "Write one new post from the BRIEF now" in user


def test_build_write_prompt_preserves_contoso_example_order_and_lineation():
    prompt = build_write_prompt(
        topic="Contoso product updates",
        points="- Explain what changed",
        examples=["First Contoso example\nwith two lines.", "Second Contoso example."],
    )

    examples_section = prompt.split(
        "EXAMPLES (rhythm reference only — names removed; never reuse their words, "
        "facts, or names):\n\n",
        1,
    )[1].split("\n\nBRIEF:", 1)[0]
    assert examples_section == (
        "First Contoso example\nwith two lines.\n\n===\n\nSecond Contoso example."
    )
