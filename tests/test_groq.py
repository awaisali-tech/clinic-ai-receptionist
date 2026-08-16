from generation.groq_client import GroqClient


def main():
    print("Testing Groq connection...")
    print("=" * 60)

    client = GroqClient()

    response = client.generate(
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "Groq connection successful."
                ),
            }
        ]
    )

    print("Response:")
    print(response)

    print("\n✓ Groq connection successful!")


if __name__ == "__main__":
    main()