from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model# Direct core import

load_dotenv()

# Initialize Mistral
model = init_chat_model(
    model="mistral-small-latest", 
    model_provider="mistralai",
    temperature=0.7 # Increased slightly for more "personality" in the modes
)

print("Choose your AI mode:")
print("1: Angry Mode 😡")
print("2: Funny Mode 😂")
print("3: Sad Mode 😭")

# Basic error handling for input
try:
    choice = int(input("Tell your response (1-3): "))
except ValueError:
    print("Invalid input, defaulting to Funny Mode.")
    choice = 2

# Map choices to system prompts
modes = {
    1: "You are an angry AI agent. You respond aggressively, use caps occasionally, and are very impatient.",
    2: "You are a very funny AI agent. You respond with puns, jokes, and a sarcastic wit.",
    3: "You are a very sad AI agent. You respond in a depressed, emotional, and sighing tone."
}

selected_mode = modes.get(choice, modes[2])

messages = [SystemMessage(content=selected_mode)]

print("\n--- Welcome! Type '0' to exit ---")

while True:
    user_input = input("You: ")
    
    if user_input == "0":
        print("Goodbye!")
        break
    
    # Add human message to history
    messages.append(HumanMessage(content=user_input))
    
    # Get AI response
    response = model.invoke(messages)
    
    # Add AI response to history
    messages.append(AIMessage(content=response.content))
    
    print(f"\nBot: {response.content}\n")