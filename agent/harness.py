from __future__ import annotations

import multiprocessing
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

import jsonschema

from agent.evidence import EvidenceRegistry
from config.settings import Settings
from llm.model import ToolCall, ToolOutcome
from tools.base import ToolRegistry, ToolResult, ToolSpec


def _process_worker(output, handler, arguments: dict[str, Any]) -> None:
    try:
        output.put(("ok", handler.run(**arguments)))
    except Exception as exc:  # process boundary serializes a bounded error
        output.put(("error", f"{type(exc).__name__}: {exc}"))


class ToolHarness:
    """Validate, execute, bound and trace all model-proposed tool calls."""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        evidence: EvidenceRegistry,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.evidence = evidence

    def execute(
        self,
        tool_call: ToolCall,
        steps: list[str],
        trace: list[dict],
    ) -> ToolOutcome:
        spec = self.registry.get(tool_call.name)
        event = {
            "type": "tool",
            "tool": tool_call.name,
            "input": tool_call.input,
            "attempts": 0,
            "ok": False,
            "duration_ms": 0.0,
            "n_sources": 0,
            "error": None,
        }
        if spec is None:
            event["error"] = "未知工具"
            trace.append(event)
            return ToolOutcome(tool_call, f"未知工具: {tool_call.name}", is_error=True)
        validation_error = self._validate(spec, tool_call.input)
        if validation_error:
            event["error"] = f"schema: {validation_error}"
            trace.append(event)
            return ToolOutcome(tool_call, f"入参不合法: {validation_error}", is_error=True)

        timeout = (
            spec.policy.timeout_seconds or self.settings.harness.tool_timeout_seconds
        )
        if spec.policy.max_retries is not None:
            retries = spec.policy.max_retries
        elif spec.policy.side_effects == "write" or not spec.policy.idempotent:
            retries = 0
        else:
            retries = self.settings.harness.tool_max_retries

        started = time.perf_counter()
        last_error = ""
        for attempt in range(1, retries + 2):
            event["attempts"] = attempt
            try:
                result = self._run_once(spec, tool_call.input, timeout)
                content, added = self.evidence.register(result)
                event.update(
                    ok=True,
                    n_sources=len(added),
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    metadata=result.metadata,
                )
                trace.append(event)
                steps.append(
                    f"Tool[{tool_call.name}] -> {len(added)} 条新增来源"
                )
                return ToolOutcome(tool_call, content)
            except TimeoutError:
                last_error = f"超时(>{timeout}s)"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        event.update(
            error=last_error,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        trace.append(event)
        steps.append(
            f"Tool[{tool_call.name}] 失败（{event['attempts']} 次）: {last_error}"
        )
        return ToolOutcome(
            tool_call,
            f"工具执行失败: {last_error}",
            is_error=True,
        )

    def _run_once(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        timeout: float,
    ) -> ToolResult:
        if spec.policy.isolate_process:
            context = multiprocessing.get_context("spawn")
            output = context.Queue(maxsize=1)
            process = context.Process(
                target=_process_worker,
                args=(output, spec.handler, arguments),
                daemon=True,
            )
            process.start()
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
                raise TimeoutError
            try:
                status, payload = output.get_nowait()
            except queue.Empty as exc:
                raise RuntimeError("isolated tool exited without a result") from exc
            if status == "error":
                raise RuntimeError(payload)
            return payload

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(spec.handler.run, **arguments)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise TimeoutError from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _validate(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
        try:
            jsonschema.validate(arguments, spec.schema["input_schema"])
            return None
        except jsonschema.ValidationError as exc:
            return exc.message
