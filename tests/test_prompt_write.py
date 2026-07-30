"""Tests for the locked RAG writing prompt."""

from personality_protect.prompt_write import build_write_prompt


def test_build_write_prompt_matches_locked_template():
    prompt = build_write_prompt(
        topic="Contoso's quarterly planning",
        points="Keep the plan focused; name one accountable owner.",
        examples=[
            "Short lines.\nClear decisions.",
            "A plan is useful only when the owner is named.",
        ],
    )

    assert prompt == (
        "System: Write a LinkedIn post in the author's voice. "
        "Match rhythm/lineation of EXAMPLES.\n"
        "Do not copy facts or names from EXAMPLES unless they appear in the BRIEF.\n"
        "No AI filler (leverage, delve, moreover, tapestry).\n"
        "\n"
        "EXAMPLES:\n"
        "Short lines.\n"
        "Clear decisions.\n"
        "\n"
        "A plan is useful only when the owner is named.\n"
        "\n"
        "BRIEF:\n"
        "Topic: Contoso's quarterly planning\n"
        "Points: Keep the plan focused; name one accountable owner.\n"
        "\n"
        "POST:"
    )


def test_build_write_prompt_preserves_contoso_example_order_and_lineation():
    prompt = build_write_prompt(
        topic="Contoso product updates",
        points="Explain what changed.",
        examples=["First Contoso example\nwith two lines.", "Second Contoso example."],
    )

    examples_section = prompt.split("EXAMPLES:\n", 1)[1].split("\n\nBRIEF:", 1)[0]
    assert examples_section == (
        "First Contoso example\nwith two lines.\n\nSecond Contoso example."
    )
