from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create the graph with MessagesState schema
builder = StateGraph(state_schema=MessagesState)

# Define the chat node function
def chat_node(state: MessagesState):
    """
    Main chatbot logic that acts as a therapy assistant.
    Stores all messages in memory.
    """
    system_message = SystemMessage(content="You're a kind therapy assistant.")
    history = state["messages"]
    prompt = [system_message] + history
    response = model.invoke(prompt)
    return {"messages": response}

# Add node to graph and set as start node
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")

# Compile graph with MemorySaver for persistence
memory = MemorySaver()
chat_app = builder.compile(checkpointer=memory)

# Unique thread identifier for this conversation
thread_id = "1"

# Main conversation loop
print("Therapy Chatbot (Basic Memory)")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    # Wrap user input as HumanMessage and update state
    state_update = {"messages": [HumanMessage(content=user_input)]}
    
    # Invoke graph with thread_id for memory persistence
    result = chat_app.invoke(
        state_update,
        {"configurable": {"thread_id": thread_id}}
    )
    
    # Extract and print AI response
    ai_msg = result["messages"][-1]
    print(f"Bot: {ai_msg.content}\n")
    
    # Uncomment below to see full message history
    # print(f"Full history: {result['messages']}\n")