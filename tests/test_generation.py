from generation.generator import Generator


def main():
    print("Testing answer generation...")
    print("=" * 60)

    generator = Generator()

    evidence = [
        """Doctor: Dr. Ayesha Khan
Specialization: Pediatrics
Clinic: Sunrise Medical Center
Experience: 10 years
Availability: Mon-Fri 9:00am-2:00pm"""
    ]

    context = {
        "doctor": "Dr. Ayesha Khan",
        "clinic": "Sunrise Medical Center",
        "specialization": "Pediatrics",
    }

    query = "When is Dr. Ayesha Khan available?"

    answer = generator.generate(
        query=query,
        evidence=evidence,
        conversation_context=context,
    )

    print("\nUser:")
    print(query)

    print("\nGenerated answer:")
    print(answer)

    print("\n✓ Generation test completed!")


if __name__ == "__main__":
    main()