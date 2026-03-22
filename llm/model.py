from __future__ import annotations

from dataclasses import dataclass

from config.settings import LLMConfig


@dataclass
class LLMOutput:
    text: str


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> LLMOutput:
        provider = self.config.provider.lower()
        if provider == "openai":
            return self._generate_openai(prompt)
        return self._generate_local(prompt)

    def _generate_openai(self, prompt: str) -> LLMOutput:
        try:
            from openai import OpenAI
        except Exception:
            return LLMOutput(text="OpenAI SDK 未安装，已回退到本地摘要模式。\n\n" + self._generate_local(prompt).text)

        client = OpenAI(api_key=self.config.openai_api_key, base_url=self.config.openai_api_base or None)
        response = client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": "你是一个专业的代码助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return LLMOutput(text=text)

    def _generate_local(self, prompt: str) -> LLMOutput:
        lines = prompt.splitlines()
        question = "未找到"
        for i, line in enumerate(lines):
            if line.strip() == "用户问题:" and i + 1 < len(lines):
                question = lines[i + 1].strip()
                break
        contexts = [line for line in lines if line.startswith("[Context")]

        summary = [
            "以下为基于检索上下文的回答（本地模式）：",
            f"问题: {question}",
            "",
            "推理结论：",
            "1. 系统先执行语义检索，再进行重排，返回最相关代码片段。",
            "2. 可从来源片段中定位关键实现位置。",
            "3. 若需更高准确度，建议接入真实大模型 API。",
            "",
            "命中上下文：",
        ]
        if contexts:
            summary.extend([f"- {line}" for line in contexts[:5]])
        else:
            summary.append("- 未检索到相关代码片段")

        return LLMOutput(text="\n".join(summary))
