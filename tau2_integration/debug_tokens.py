
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tau2_integration.tracing import TraceRecorder
from tau2_integration.callbacks import TracingCallbackHandler
import uuid
from dotenv import load_dotenv

load_dotenv()

def debug_token_capture():
    print("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    recorder = TraceRecorder(task_id="debug_tokens", framework="test")
    handler = TracingCallbackHandler(recorder)
    
    print("Invoking LLM...")
    try:
        # We need to manually trigger the callback logic or use the handler with the LLM
        # Attaching handler directly to invoke config
        response = llm.invoke(
            [HumanMessage(content="Hello, say 'test'")],
            config={"callbacks": [handler]}
        )
        
        print("\n--- LLM Response Object ---")
        print(f"Content: {response.content}")
        print(f"Response Metadata: {response.response_metadata}")
        print(f"Usage Metadata: {response.usage_metadata}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_token_capture()
