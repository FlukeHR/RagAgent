import { describe, expect, test } from "vitest";

import { safeMarkdown, sourceForKey } from "./render.js";


describe("safeMarkdown", () => {
  test("removes active content and preserves plain answer text", () => {
    const result = safeMarkdown('结论<script>alert(1)</script><img src=x onerror="alert(2)">');
    expect(result.textContent).toContain("结论");
    expect(result.querySelector("script")).toBeNull();
    expect(result.querySelector("img")?.getAttribute("onerror")).toBeNull();
  });

  test("creates an external citation link backed by structured source data", () => {
    const source = {
      id: "S1",
      paper_title: "Demo",
      section: "Abstract",
      source_kind: "external_url",
      citation_url: "https://arxiv.org/abs/2601.00001",
    };
    const result = safeMarkdown("外部结论 [S1]。", [source]);
    const citation = result.querySelector("a.citation");
    expect(citation?.getAttribute("href")).toBe(source.citation_url);
    expect(sourceForKey(citation.dataset.citationKey)).toEqual(source);
  });

  test("does not convert citations inside code", () => {
    const result = safeMarkdown("`[S1]`", [{ id: "S1", source_kind: "library_pdf" }]);
    expect(result.querySelector(".citation")).toBeNull();
  });
});
