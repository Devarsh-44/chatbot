import time
from backend import F1KnowledgeBase

# Initialize the backend
kb = F1KnowledgeBase(user_login="example_user")

# Streamed response emulator
def response_generator(prompt):
    response = kb.generate_response(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    print("Hamilton Chatbot")
    chat_history = []

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        response = "".join(response_generator(prompt))
        print(f"Hamilton: {response}")

        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()