from langchain_core.messages import HumanMessage

from langchain_agent import lauki_agent


def main() -> None:
    """Run the Lauki Phones agent in an interactive terminal session."""
    messages = []
    print("Lauki Phones assistant. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        response = lauki_agent.invoke({"messages": messages})
        messages = response["messages"]
        print(f"Assistant: {messages[-1].content}")


if __name__ == "__main__":
    main()
