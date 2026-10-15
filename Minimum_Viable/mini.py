"""最小可用的 Agent Harness —— 单文件。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# ═══════════ 第 1 层：工具 ═══════════

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]      # JSON Schema
    fn: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool
        return self                    # 链式注册，写起来爽

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[error] 未注册的工具: {name}"
        try:
            return str(tool.fn(**arguments))
        except Exception as exc:
            # 关键设计：工具报错不炸循环，而是作为"观察结果"回喂给模型
            # 模型看到报错会自己改参数重试 —— 这才是 agent 的纠错能力来源
            return f"[error] {type(exc).__name__}: {exc}"

# ═══════════ 第 2 层：模型适配 ═══════════

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class Reply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

class Adapter(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...

class OpenAICompatAdapter:
    """适配任何 OpenAI 兼容端点：vLLM / Ollama / OpenAI / DeepSeek…"""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", **kw):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.kw = kw

    @staticmethod
    def _parse(raw: str | None) -> dict[str, Any]:
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            **self.kw,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name,
                     arguments=self._parse(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        return Reply(content=msg.content, tool_calls=calls)

# ═══════════ 第 3 层：循环控制 ═══════════

@dataclass
class Trace:
    messages: list[dict[str, Any]]
    stop_reason: str = "running"
    steps: int = 0

    @property
    def final(self) -> str | None:
        return self.messages[-1]["content"] if self.stop_reason == "completed" else None

class Harness:
    def __init__(self, adapter: Adapter, registry: ToolRegistry,
                 system_prompt: str = "", max_steps: int = 12) -> None:
        self.adapter = adapter
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_input: str) -> Trace:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})

        trace = Trace(messages=messages)
        schemas = self.registry.schemas()

        while trace.steps < self.max_steps:
            trace.steps += 1
            reply = self.adapter.chat(messages, schemas)

            # 终止条件 A：模型不再请求工具 —— 它认为可以交卷了
            if not reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.content or ""})
                trace.stop_reason = "completed"
                return trace

            # 把 assistant 的 tool_calls 落进历史（OpenAI 协议强制要求，`
            # 否则下一轮 tool 消息找不到对应的 tool_call_id，请求会被拒）
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name,
                                  "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                    for c in reply.tool_calls
                ],
            })

            for call in reply.tool_calls:        # 天然支持并行工具调用
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": self.registry.execute(call.name, call.arguments),
                })

        trace.stop_reason = "max_steps_exceeded"
        return trace

# ═══════════ 用法 ═══════════

if __name__ == "__main__":
    tools = (
        ToolRegistry()
        .register(Tool(
            name="calc",
            description="计算一个 Python 算术表达式，例如 '(3+5)*2'",
            parameters={"type": "object",
                        "properties": {"expr": {"type": "string", "description": "算术表达式"}},
                        "required": ["expr"]},
            fn=lambda expr: eval(expr, {"__builtins__": {}}, {}),   # 演示用，别上生产
        ))
        .register(Tool(
            name="write_file",
            description="把内容写入指定文件",
            parameters={"type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]},
            fn=lambda path, content: (open(path, "w", encoding="utf-8").write(content), "ok")[1],
        ))
    )

    harness = Harness(
        adapter=OpenAICompatAdapter(
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://localhost:8000/v1",   # 本地 vLLM 的 OpenAI 兼容口
            temperature=0.0,                       # agent 场景必须低温，否则工具名会飘
        ),
        registry=tools,
        system_prompt="你是会使用工具的助手。需要计算或写文件时请调用工具，不要凭记忆回答。",
        max_steps=8,
    )

    trace = harness.run("帮我算一下 (17*23+9)/4，然后把结果写进 D:/tmp/out.txt")
    print("停止原因:", trace.stop_reason, "| 步数:", trace.steps)
    print("最终回答:", trace.final)