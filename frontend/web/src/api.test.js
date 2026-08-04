import { afterEach, describe, expect, it, vi } from "vitest";

import { jsonBody, setCsrfToken, streamApi } from "./api.js";

describe("streamApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    setCsrfToken("");
  });

  it("parses split NDJSON token and final events", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode('{"type":"token","text":"快')); 
        controller.enqueue(encoder.encode('速"}\n{"type":"reset"}\n'));
        controller.enqueue(encoder.encode('{"type":"final","result":{"answer":"快速"}}\n'));
        controller.close();
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(body, {
        status: 200,
        headers: { "Content-Type": "application/x-ndjson" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    setCsrfToken("csrf-token");
    const events = [];

    await streamApi(
      "/sessions/id/ask/stream",
      { method: "POST", body: jsonBody({ question: "q" }) },
      (event) => events.push(event),
    );

    expect(events.map((event) => event.type)).toEqual(["token", "reset", "final"]);
    expect(events[0].text).toBe("快速");
    expect(events[2].result.answer).toBe("快速");
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers.get("X-CSRF-Token")).toBe("csrf-token");
  });
});
