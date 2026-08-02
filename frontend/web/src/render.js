import DOMPurify from "dompurify";
import { marked } from "marked";

const citationRegistry = new Map();
let citationCounter = 0;

marked.setOptions({ gfm: true, breaks: true });

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);
}

export function safeMarkdown(text, sources = []) {
  const wrapper = document.createElement("div");
  const dirty = marked.parse(String(text || ""));
  wrapper.innerHTML = DOMPurify.sanitize(dirty, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "iframe", "object", "embed", "form", "input", "button"],
    FORBID_ATTR: ["style", "srcdoc"],
  });
  const sourceMap = new Map(sources.map((source, index) => [
    normalizeSourceId(source.id, index), source,
  ]));
  const walker = document.createTreeWalker(wrapper, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const pattern = /\[(S\s*\d+)\]/gi;
  for (const node of nodes) {
    if (node.parentElement?.closest("code, pre, a")) continue;
    const text = node.nodeValue || "";
    let match;
    let cursor = 0;
    const fragment = document.createDocumentFragment();
    let changed = false;
    while ((match = pattern.exec(text))) {
      const sourceId = match[1].toUpperCase().replace(/\s+/g, "");
      const source = sourceMap.get(sourceId);
      if (!source) continue;
      changed = true;
      fragment.append(text.slice(cursor, match.index));
      fragment.append(createCitation(sourceId, source));
      cursor = pattern.lastIndex;
    }
    if (changed) {
      fragment.append(text.slice(cursor));
      node.replaceWith(fragment);
    }
  }
  return wrapper;
}

export function sourcePills(sources = []) {
  const container = document.createElement("div");
  container.className = "source-pills";
  if (!sources.length) return container;
  const label = document.createElement("span");
  label.className = "source-label";
  label.textContent = "参考来源";
  container.append(label);
  sources.forEach((source, index) => {
    container.append(createCitation(normalizeSourceId(source.id, index), source));
  });
  return container;
}

export function sourceForKey(key) {
  return citationRegistry.get(key) || null;
}

function normalizeSourceId(value, index) {
  const match = String(value || "").toUpperCase().match(/S\s*(\d+)/);
  return match ? `S${match[1]}` : `S${index + 1}`;
}

function createCitation(id, source) {
  const key = `cite-${++citationCounter}`;
  citationRegistry.set(key, source);
  const external = source.source_kind === "external_url" && source.citation_url;
  const element = document.createElement(external ? "a" : "button");
  element.className = `citation ${external ? "citation-external" : "citation-local"}`;
  element.dataset.citationKey = key;
  element.title = `${source.paper_title || "来源"}${source.section ? ` · ${source.section}` : ""}`;
  element.setAttribute("aria-label", `预览来源 ${id}`);
  if (external) {
    element.href = source.citation_url;
    element.target = "_blank";
    element.rel = "noopener noreferrer";
  } else {
    element.type = "button";
  }
  element.textContent = id;
  return element;
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(Number(value) * 1000);
  return Number.isNaN(date.getTime()) ? "—" : new Intl.DateTimeFormat("zh-CN", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

export function statusLabel(status) {
  return ({
    queued: "等待处理",
    parsing: "正在解析",
    indexing: "正在索引",
    ready: "可检索",
    succeeded: "已完成",
    failed: "失败",
    deleting: "正在删除",
  })[status] || status || "未知";
}
