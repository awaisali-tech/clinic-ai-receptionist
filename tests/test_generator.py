from generation.generator import Generator


def main():

    print("Testing Generator...")
    print("=" * 60)

    generator = Generator()

    answer = generator.generate(
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

    print("\nGenerated answer:")
    print("=" * 60)
    print(answer)

    assert answer
    assert isinstance(answer, str)

    print("\n✓ Generator test passed!")


if __name__ == "__main__":
    main()