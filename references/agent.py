from typing import List, Dict, Optional, Callable
from pydantic import BaseModel
from message.message import llmMessage, Turn, adapt_messages_for_chat_template
from model.llm import BaseLLM
from string import Template
import requests
import re
import time


class BaseAgent:

    def no_memory_step(self):
        pass

    def step(self):
        pass

    def init_system_prompt(self, template: str, args: dict):
        pass

    def add_memory(self):
        pass

    def reset(self):
        pass


class Agent(BaseAgent):
    llm: BaseLLM = None
    prompt_template: str = ""
    system_prompt: llmMessage = llmMessage(role="system", content="")
    memory: List[llmMessage] = []
    name: str = ""

    def step(self) -> llmMessage:
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        response = self.llm.generate_response(message_input, self.name)

        self.add_memory(response)

        return response

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)


def _is_iteration_0(system_prompt_content: str) -> bool:
    # At iteration 0 prompts carry the explicit "begin your response with X:"
    # instruction, so the model starts its own turn with the name and no
    # prefill is needed. Later iterations rely on the prefill mechanism.
    return (
        "You should start your utterance with" in system_prompt_content
        or "You must begin your response with" in system_prompt_content
    )


class VllmAgent(BaseAgent):
    """
    The agent class is based on VLLM.
    It handles communication , manages the conversation context (memory),
    and formats the input/output in the required structure.

    Model-agnostic: no hardcoded chat template. Name-prefix continuation for
    non-iteration-0 turns is done with the OpenAI-compatible
    ``continue_final_message`` + ``add_generation_prompt: false`` prefill,
    which vLLM >= 0.6.3 (with transformers >= 4.45) supports: the request
    ends with an unterminated assistant message "Alice:" and the response
    content is the natural continuation (the prefill itself is NOT echoed
    back), so the agent prepends the name prefix itself.

    For Gemma-style chat templates (which reject the system role), pass
    ``merge_system_into_user=True``: the outgoing message list is adapted
    (system prompt merged into the first user message, roles alternate
    starting with user) before the request; the recorded system prompt and
    memory are left untouched.
    """

    def __init__(
        self,
        url: str,
        my_model_name: str,
        name: str,
        temperature: float,
        max_tokens: int = 2000,
        seed_provider: Optional[Callable[[], Optional[int]]] = None,
        use_name_prefix: bool = True,
        merge_system_into_user: bool = False,
    ):
        self.url = url
        self.prompt_template = ""
        self.system_prompt: llmMessage = llmMessage(role="system", content="")
        self.memory: List[llmMessage] = []
        self.my_model_name = my_model_name
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # deterministic sampling: returns a per-request seed or None
        self.seed_provider = seed_provider
        self.use_name_prefix = use_name_prefix
        self.merge_system_into_user = merge_system_into_user

    def init_system_prompt(self, template: str, args: dict):
        self.system_prompt.content = Template(template).safe_substitute(args)
        self.memory.append(self.system_prompt)

    def add_memory(self, new_memory: llmMessage):
        self.memory.append(new_memory)

    def reset(self):
        self.memory = []
        self.system_prompt = llmMessage(role="system", content=self.prompt_template)

    def _request(self) -> Turn:
        message_input = [
            {"role": message.role, "content": message.content}
            for message in self.memory
        ]
        if self.merge_system_into_user:
            message_input = adapt_messages_for_chat_template(message_input, True)
        is_iteration_0 = _is_iteration_0(self.system_prompt.content)
        headers = {"Content-Type": "application/json"}
        data_json = {
            "model": self.my_model_name,
            "messages": list(message_input),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed_provider is not None:
            seed = self.seed_provider()
            if seed is not None:
                data_json["seed"] = seed
        use_prefill = self.use_name_prefix and not is_iteration_0
        if use_prefill:
            # prefill the assistant turn with the name prefix and ask vLLM
            # to continue exactly that final message instead of starting a
            # fresh assistant turn.
            data_json["messages"].append(
                {"role": "assistant", "content": f"{self.name}:"}
            )
            data_json["continue_final_message"] = True
            data_json["add_generation_prompt"] = False

        response = requests.post(self.url, headers=headers, json=data_json, timeout=600)
        if response.status_code == 400:
            return Turn(
                role="assistant", content="error", speaker=self.name,
                finish_reason="error",
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM endpoint {self.url} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        payload = response.json()
        choice = payload["choices"][0]
        content: str = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "") or ""
        if use_prefill and not content.startswith(self.name):
            # prefill part is not echoed back; restore the name prefix
            content = f"{self.name}:{content}"
        token_count = 0
        try:
            token_count = int(payload["usage"]["completion_tokens"])
        except (KeyError, TypeError, ValueError):
            token_count = len(content.split())

        return Turn(
            role="assistant",
            content=content,
            speaker=self.name,
            token_count=token_count,
            finish_reason=finish_reason,
        )

    # step and update memory
    def step(self) -> Turn:
        last_error: Optional[Exception] = None
        for attempt in range(3):  # 1 initial attempt + 2 retries
            try:
                response = self._request()
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(5)
        else:
            print(f"[agent] {self.name} step failed after retries: {last_error}")
            response = Turn(
                role="assistant", content="error", speaker=self.name,
                finish_reason="error",
            )
        if response.content != "error":  # keep "error" out of the context, like the old 400 path
            self.add_memory(response)
        return response

    # step but don't update memory
    def no_memory_step(self) -> Turn:
        old_memory = list(self.memory)
        response = self.step()
        self.memory = old_memory
        return response
