from conversation.entity_resolver import ContextResolver


def main():

    resolver = ContextResolver()

    # -----------------------------------------
    # Turn 1
    # -----------------------------------------

    resolver.update(
        doctor="Dr. Ayesha Khan",
        clinic="Sunrise Medical Center",
    )

    result = resolver.resolve(
        "When is Dr. Ayesha Khan available?"
    )

    print("Turn 1:")
    print(result)

    # -----------------------------------------
    # Turn 2
    # -----------------------------------------

    result = resolver.resolve(
        "What about Saturday?"
    )

    print("\nTurn 2:")
    print(result)

    # -----------------------------------------
    # Turn 3
    # -----------------------------------------

    result = resolver.resolve(
        "What time?"
    )

    print("\nTurn 3:")
    print(result)

    # -----------------------------------------
    # Reset
    # -----------------------------------------

    result = resolver.resolve(
        "I have a different question"
    )

    print("\nAfter reset:")
    print(result)


if __name__ == "__main__":
    main()