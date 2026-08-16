from generation.prompt_builder import PromptBuilder


def main():

    builder = PromptBuilder()

    messages = builder.build(
        query="When is Dr. Ayesha Khan available?",
        evidence=[
            (
                "Doctor: Dr. Ayesha Khan\n"
                "Specialization: Pediatrics\n"
                "Clinic: Sunrise Medical Center\n"
                "Experience: 10 years\n"
                "Availability: Mon-Fri 9:00am-2:00pm"
            )
        ],
        conversation_context={
            "doctor": "Dr. Ayesha Khan",
            "clinic": "Sunrise Medical Center",
            "specialization": "Pediatrics",
        },
    )

    print("System prompt:")
    print("=" * 60)
    print(messages[0]["content"])

    print("\nUser prompt:")
    print("=" * 60)
    print(messages[1]["content"])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    assert "Dr. Ayesha Khan" in messages[1]["content"]
    assert "Sunrise Medical Center" in messages[1]["content"]
    assert "Mon-Fri 9:00am-2:00pm" in messages[1]["content"]

    print("\n✓ Prompt builder test passed!")


if __name__ == "__main__":
    main()