let csrfToken = "";

export function setCsrfToken(value) {
  csrfToken = value || "";
}

export function getCsrfToken() {
  return csrfToken;
}

export async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...options,
    method,
    headers,
  });
  if (response.status === 204) return null;
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof payload === "object" ? payload.detail : payload;
    const error = new Error(message || `请求失败（${response.status}）`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

export function jsonBody(value) {
  return JSON.stringify(value);
}
