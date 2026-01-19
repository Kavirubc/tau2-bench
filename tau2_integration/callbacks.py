from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .tracing import TraceRecorder

class TracingCallbackHandler(BaseCallbackHandler):
    """Callback handler to feed LangChain events into TraceRecorder."""
    
    def __init__(self, recorder: TraceRecorder):
        self.recorder = recorder
        self.step_map: Dict[str, str] = {}  # Map run_id -> step_id
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        step_id = self.recorder.start_step(
            step_type="llm",
            name="llm_generation",
            input_data=prompts[0] if prompts else "",
            metadata={"model": kwargs.get("invocation_params", {}).get("model_name")}
        )
        self.step_map[str(run_id)] = step_id
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        """Handle chat model start."""
        try:
            # messages is a list of lists of BaseMessage
            input_str = ""
            if messages and messages[0]:
                try:
                    input_str = str(messages[0]) # naive stringification for now
                except Exception:
                    input_str = "Error stringifying messages"
                
            step_id = self.recorder.start_step(
                step_type="llm", # Use same type for consistency
                name="chat_generation",
                input_data=input_str,
                metadata={"model": kwargs.get("invocation_params", {}).get("model_name")}
            )
            self.step_map[str(run_id)] = step_id
        except Exception:
            pass

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        step_id = self.step_map.pop(str(run_id), None)
        if step_id:
            # Extract token usage
            token_usage = {}
            # Search for token usage in multiple locations
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                token_usage = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
            elif hasattr(response, "usage_metadata") and response.usage_metadata:
                 # Top-level usage_metadata
                 usage = response.usage_metadata
                 token_usage = {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
            elif hasattr(response, "generations") and response.generations:
                # Try getting from generation info
                gen = response.generations[0][0]
                if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                    usage = gen.message.usage_metadata
                    token_usage = {
                        "input": usage.get("input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "total": usage.get("total_tokens", 0)
                    }
            
            self.recorder.end_step(
                step_id=step_id, 
                output=response.generations[0][0].text,
                tokens=token_usage
            )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        step_id = self.step_map.pop(str(run_id), None)
        if step_id:
            self.recorder.end_step(step_id, output=None, error=str(error))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        name = serialized.get("name") if serialized else "unknown_tool"
        step_id = self.recorder.start_step(
            step_type="tool",
            name=name,
            input_data=input_data(kwargs.get("inputs", {}) or input_str),
        )
        self.step_map[str(run_id)] = step_id
    def on_tool_end(
        self, output: str, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        step_id = self.step_map.pop(str(run_id), None)
        if step_id:
            self.recorder.end_step(step_id, output=output)

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> Any:
        step_id = self.step_map.pop(str(run_id), None)
        if step_id:
            self.recorder.end_step(step_id, output=None, error=str(error))

def input_data(data: Any) -> str:
    """Helper to format input data."""
    if isinstance(data, dict):
        return str(data)
    return str(data)
