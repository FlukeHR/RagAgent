import "./styles.css";
import { api, jsonBody, setCsrfToken } from "./api.js";
import {
  escapeHtml,
  formatDate,
  safeMarkdown,
  sourceForKey,
  sourcePills,
  statusLabel,
} from "./render.js";

const app = document.querySelector("#app");
const state = {
  auth: null,
  authMode: "login",
  route: "overview",
  dashboard: null,
  profiles: [],
  sessions: [],
  activeSession: null,
  papers: [],
  jobs: [],
  candidates: [],
  proposalId: null,
  viewer: null,
  viewerTab: "preview",
  editProfileId: null,
  sending: false,
  mobileNav: false,
};

let pollTimer = null;
let previewNonce = 0;

bootstrap();
document.addEventListener("click", citationClick, true);

async function bootstrap() {
  state.route = routeFromHash();
  try {
    const auth = await api("/auth/session");
    setAuthenticated(auth);
    await loadCommon();
    await loadRoute();
    maybeOfferLegacyImport();
  } catch (error) {
    if (error.status !== 401) toast(error.message, true);
    renderAuth();
  }
}

function setAuthenticated(auth) {
  state.auth = auth;
  setCsrfToken(auth.csrf_token);
}

function routeFromHash() {
  const value = location.hash.replace(/^#\/?/, "").split("/")[0];
  return ["overview", "chat", "library", "settings"].includes(value) ? value : "overview";
}

window.addEventListener("hashchange", async () => {
  state.route = routeFromHash();
  state.mobileNav = false;
  if (!state.auth) return;
  try {
    await loadRoute();
  } catch (error) {
    toast(error.message, true);
  }
});

async function loadCommon() {
  const [profiles, sessions] = await Promise.all([
    api("/model-profiles"),
    api("/sessions"),
  ]);
  state.profiles = profiles;
  state.sessions = sessions;
}

async function loadRoute() {
  stopPolling();
  if (state.route === "overview") {
    state.dashboard = await api("/dashboard");
  } else if (state.route === "chat") {
    await loadCommon();
    if (state.activeSession?.conversation_id) {
      state.activeSession = await api(`/sessions/${state.activeSession.conversation_id}`);
    } else if (state.sessions.length) {
      state.activeSession = await api(`/sessions/${state.sessions[0].conversation_id}`);
    }
  } else if (state.route === "library") {
    await loadLibrary();
    startPolling();
  } else if (state.route === "settings") {
    await loadCommon();
  }
  renderShell();
}

function renderAuth() {
  stopPolling();
  state.auth = null;
  app.innerHTML = `
    <main class="auth-shell">
      <section class="auth-story">
        <div class="brand"><span class="brand-mark">P</span><span>PaperDesk</span></div>
        <div class="auth-copy">
          <div class="eyebrow">Local-first research workspace</div>
          <h1>让每篇论文，都能成为可追溯的答案。</h1>
          <p>在本机管理论文、解析全文、建立检索索引，并与自己的模型安全对话。数据与密钥不会离开你控制的设备。</p>
        </div>
        <div class="auth-foot">MinerU · Agentic RAG · 可回查引用</div>
      </section>
      <section class="auth-panel">
        <div class="auth-card">
          <h2>${state.authMode === "login" ? "欢迎回来" : "创建本地账户"}</h2>
          <p>${state.authMode === "login" ? "继续你的论文研究工作。" : "不同账户的论文、索引和模型密钥完全隔离。"}</p>
          <div class="auth-switch">
            <button type="button" data-auth-mode="login" class="${state.authMode === "login" ? "active" : ""}">登录</button>
            <button type="button" data-auth-mode="register" class="${state.authMode === "register" ? "active" : ""}">注册</button>
          </div>
          <form id="auth-form">
            ${state.authMode === "register" ? `
              <div class="field"><label for="display-name">显示名称</label><input id="display-name" name="display_name" autocomplete="name" maxlength="80" placeholder="例如：林同学" /></div>
            ` : ""}
            <div class="field"><label for="username">用户名</label><input id="username" name="username" autocomplete="username" minlength="3" maxlength="32" required placeholder="字母、数字、点、横线或下划线" /></div>
            <div class="field"><label for="password">密码</label><input id="password" name="password" type="password" autocomplete="${state.authMode === "login" ? "current-password" : "new-password"}" minlength="10" maxlength="256" required /></div>
            <div id="auth-error" class="form-error"></div>
            <button class="primary wide" type="submit">${state.authMode === "login" ? "登录工作台" : "创建并登录"}</button>
          </form>
        </div>
      </section>
    </main>`;
  document.querySelectorAll("[data-auth-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.authMode = button.dataset.authMode;
      renderAuth();
    });
  });
  document.querySelector("#auth-form").addEventListener("submit", submitAuth);
}

async function submitAuth(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  document.querySelector("#auth-error").textContent = "";
  try {
    const payload = Object.fromEntries(form.entries());
    const result = await api(`/auth/${state.authMode}`, {
      method: "POST", body: jsonBody(payload),
    });
    setAuthenticated(result);
    state.route = "overview";
    location.hash = "#/overview";
    await loadCommon();
    await loadRoute();
    maybeOfferLegacyImport();
  } catch (error) {
    document.querySelector("#auth-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function renderShell() {
  const user = state.auth.user;
  app.innerHTML = `
    <div class="app-shell">
      <aside class="sidebar ${state.mobileNav ? "mobile-open" : ""}" id="sidebar">
        <div class="brand"><span class="brand-mark">P</span><span>PaperDesk</span></div>
        <nav class="nav" aria-label="主导航">
          ${navItem("overview", "01", "总览")}
          ${navItem("chat", "02", "对话")}
          ${navItem("library", "03", "论文库")}
          ${navItem("settings", "04", "设置")}
        </nav>
        <div class="sidebar-bottom">
          <div class="user-chip">
            <div class="avatar">${escapeHtml((user.display_name || user.username).slice(0, 1).toUpperCase())}</div>
            <div class="user-meta"><div class="user-name">${escapeHtml(user.display_name)}</div><div class="user-sub">@${escapeHtml(user.username)}</div></div>
            <button class="ghost" id="quick-logout" title="退出登录">退出</button>
          </div>
        </div>
      </aside>
      <main class="main-shell">
        <header class="topbar">
          <button class="ghost menu-button" id="mobile-menu" aria-label="打开导航">菜单</button>
          <h1>${pageName()}</h1><div class="spacer"></div>
          <span class="status-dot" aria-hidden="true"></span><span class="top-note">本地服务</span>
        </header>
        ${renderRoute()}
      </main>
      <div class="viewer-backdrop ${state.viewer ? "open" : ""}" id="viewer-backdrop"></div>
      <aside class="viewer ${state.viewer ? "open" : ""}" id="viewer" aria-label="来源预览"></aside>
    </div>`;
  bindShell();
  renderViewer();
}

function navItem(route, icon, label) {
  return `<button class="nav-item ${state.route === route ? "active" : ""}" data-route="${route}"><span class="nav-icon">${icon}</span><span>${label}</span></button>`;
}

function pageName() {
  return ({ overview: "研究总览", chat: "论文对话", library: "我的论文库", settings: "账户与模型" })[state.route];
}

function renderRoute() {
  if (state.route === "overview") return renderOverview();
  if (state.route === "chat") return renderChat();
  if (state.route === "library") return renderLibrary();
  return renderSettings();
}

function bindShell() {
  document.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => { location.hash = `#/${button.dataset.route}`; });
  });
  document.querySelector("#mobile-menu")?.addEventListener("click", () => {
    state.mobileNav = !state.mobileNav;
    document.querySelector("#sidebar")?.classList.toggle("mobile-open", state.mobileNav);
  });
  document.querySelector("#quick-logout")?.addEventListener("click", logout);
  document.querySelector("#viewer-backdrop")?.addEventListener("click", closeViewer);
  bindRoute();
}

function bindRoute() {
  if (state.route === "overview") bindOverview();
  else if (state.route === "chat") bindChat();
  else if (state.route === "library") bindLibrary();
  else bindSettings();
}

function renderOverview() {
  const data = state.dashboard || { papers: {}, recent_conversations: [], recent_jobs: [] };
  const papers = data.papers || {};
  const total = Object.values(papers).reduce((sum, value) => sum + Number(value), 0);
  return `<section class="page"><div class="content-width">
    <div class="hero-panel">
      <div><div class="eyebrow">YOUR LOCAL RESEARCH DESK</div><h2>从论文库出发，得到有出处的回答。</h2><p>上传论文或从 arXiv 选择候选，MinerU 会在后台解析并建立你的私有检索索引。</p></div>
      <div class="hero-actions"><button class="primary" data-route="chat">开始提问</button><button class="secondary" data-route="library">添加论文</button></div>
    </div>
    <div class="stats-grid">
      ${statCard("论文总数", total, `${papers.ready || 0} 篇可检索`)}
      ${statCard("正在处理", data.active_jobs || 0, "单 worker 有序执行")}
      ${statCard("失败项目", papers.failed || 0, "可在论文库中重试")}
      ${statCard("模型配置", data.model_profiles || 0, "密钥仅加密保存在本机")}
    </div>
    <div class="overview-grid">
      ${activityPanel("最近对话", data.recent_conversations || [], "conversation")}
      ${activityPanel("入库动态", data.recent_jobs || [], "job")}
    </div>
  </div></section>`;
}

function statCard(label, value, detail) {
  return `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-detail">${detail}</div></div>`;
}

function activityPanel(title, items, kind) {
  const rows = items.length ? items.map((item) => kind === "conversation"
    ? `<button class="activity-row ghost wide" data-open-session="${item.conversation_id}"><div class="activity-main"><div class="activity-title">${escapeHtml(item.title)}</div><div class="activity-meta">${item.message_count || 0} 条消息 · ${formatDate(item.updated_at)}</div></div></button>`
    : `<div class="activity-row"><span class="status ${item.status}">${statusLabel(item.status)}</span><div class="activity-main"><div class="activity-title">任务 ${escapeHtml(String(item.job_id).slice(0, 8))}</div><div class="activity-meta">${formatDate(item.updated_at)}</div></div></div>`).join("")
    : `<div class="empty-state">这里还没有记录。<br />上传一篇论文或开始一次对话吧。</div>`;
  return `<section class="panel"><div class="panel-head"><h3>${title}</h3></div><div class="panel-body">${rows}</div></section>`;
}

function bindOverview() {
  document.querySelectorAll("[data-route]").forEach((button) => button.addEventListener("click", () => { location.hash = `#/${button.dataset.route}`; }));
  document.querySelectorAll("[data-open-session]").forEach((button) => button.addEventListener("click", async () => {
    state.activeSession = await api(`/sessions/${button.dataset.openSession}`);
    location.hash = "#/chat";
  }));
}

function renderChat() {
  const session = state.activeSession;
  return `<section class="chat-layout">
    <aside class="session-rail"><div class="session-head"><button class="secondary wide" id="new-session">＋ 新建对话</button></div><div class="session-list">
      ${state.sessions.length ? state.sessions.map((item) => `<div class="session-item ${session?.conversation_id === item.conversation_id ? "active" : ""}"><button class="session-select" data-session="${item.conversation_id}"><span class="session-title">${escapeHtml(item.title)}</span><span class="session-time">${formatDate(item.updated_at)}</span></button><button class="ghost session-delete" data-delete-session="${item.conversation_id}" aria-label="删除对话">×</button></div>`).join("") : `<div class="empty-state">还没有对话</div>`}
    </div></aside>
    <div class="chat-stage">
      <div class="chat-head">
        <button class="ghost" id="mobile-new-session">＋</button>
        <div class="chat-title"><strong>${escapeHtml(session?.title || "选择或新建对话")}</strong><span>${session ? `${(session.messages || []).length} 条消息` : "选择模型配置后开始提问"}</span></div>
        <select class="model-select" id="chat-model" ${session ? "" : "disabled"} aria-label="当前模型配置">
          <option value="">选择模型配置</option>${state.profiles.map((profile) => `<option value="${profile.profile_id}" ${session?.model_profile_id === profile.profile_id ? "selected" : ""}>${escapeHtml(profile.name)} · ${escapeHtml(profile.model_name)}</option>`).join("")}
        </select>
      </div>
      <div class="messages" id="messages"></div>
      <div class="composer-wrap"><form class="composer" id="composer"><textarea id="question" rows="1" maxlength="20000" placeholder="向你的论文库提问……" ${!session || !session.model_profile_id ? "disabled" : ""}></textarea><button class="send-button" type="submit" ${!session || !session.model_profile_id || state.sending ? "disabled" : ""}>↑</button></form><div class="composer-hint">Enter 发送，Shift + Enter 换行。回答中的引用可在右侧回查。</div></div>
    </div>
  </section>`;
}

function bindChat() {
  renderMessages();
  document.querySelector("#new-session")?.addEventListener("click", newSession);
  document.querySelector("#mobile-new-session")?.addEventListener("click", newSession);
  document.querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", async () => {
    state.activeSession = await api(`/sessions/${button.dataset.session}`);
    renderShell();
  }));
  document.querySelectorAll("[data-delete-session]").forEach((button) => button.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!await confirmModal("删除对话？", "该对话及其中的消息将被永久删除。")) return;
    await api(`/sessions/${button.dataset.deleteSession}`, { method: "DELETE" });
    state.activeSession = null;
    await loadCommon();
    if (state.sessions.length) state.activeSession = await api(`/sessions/${state.sessions[0].conversation_id}`);
    renderShell();
  }));
  document.querySelector("#chat-model")?.addEventListener("change", async (event) => {
    if (!state.activeSession) return;
    state.activeSession = await api(`/sessions/${state.activeSession.conversation_id}`, {
      method: "PATCH", body: jsonBody({ model_profile_id: event.target.value || null }),
    });
    await loadCommon();
    renderShell();
  });
  document.querySelector("#composer")?.addEventListener("submit", sendQuestion);
  const question = document.querySelector("#question");
  question?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); question.form.requestSubmit(); }
  });
  question?.addEventListener("input", () => {
    question.style.height = "auto";
    question.style.height = `${Math.min(question.scrollHeight, 150)}px`;
    localStorage.setItem("paperdesk_draft", question.value);
  });
  if (question) question.value = localStorage.getItem("paperdesk_draft") || "";
}

function renderMessages() {
  const target = document.querySelector("#messages");
  if (!target) return;
  const messages = state.activeSession?.messages || [];
  if (!messages.length && !state.sending) {
    target.innerHTML = `<div class="empty-state"><strong>从一个具体问题开始</strong><br />例如：“这篇论文的实验设置与基线有哪些差异？”</div>`;
    return;
  }
  target.replaceChildren();
  messages.forEach((message) => {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;
    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "你" : "PaperDesk";
    article.append(role);
    if (message.role === "user") {
      const bubble = document.createElement("div");
      bubble.className = "user-bubble";
      bubble.textContent = message.content;
      article.append(bubble);
    } else {
      const bubble = safeMarkdown(message.content, message.sources || []);
      bubble.className = "assistant-bubble";
      article.append(bubble, sourcePills(message.sources || []));
      if (message.steps?.length) {
        const details = document.createElement("details");
        details.className = "steps";
        details.innerHTML = `<summary>查看检索过程（${message.steps.length} 步）</summary><ul>${message.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>`;
        article.append(details);
      }
    }
    target.append(article);
  });
  if (state.sending) {
    const thinking = document.createElement("article");
    thinking.className = "message assistant";
    thinking.innerHTML = `<div class="message-role">PaperDesk</div><div class="thinking">正在规划检索并核对引用</div>`;
    target.append(thinking);
  }
  target.scrollTop = target.scrollHeight;
  window.MathJax?.typesetPromise?.([target]).catch(() => {});
}

async function newSession() {
  const profile = state.profiles.find((item) => item.is_default) || state.profiles[0];
  if (!profile) { toast("请先在设置中添加模型配置", true); location.hash = "#/settings"; return; }
  state.activeSession = await api("/sessions", {
    method: "POST", body: jsonBody({ title: "新对话", model_profile_id: profile.profile_id }),
  });
  await loadCommon();
  renderShell();
  document.querySelector("#question")?.focus();
}

async function sendQuestion(event) {
  event.preventDefault();
  if (!state.activeSession || state.sending) return;
  const input = document.querySelector("#question");
  const question = input.value.trim();
  if (!question) return;
  state.sending = true;
  state.activeSession.messages.push({ role: "user", content: question, sources: [] });
  input.value = "";
  localStorage.removeItem("paperdesk_draft");
  renderMessages();
  try {
    await api(`/sessions/${state.activeSession.conversation_id}/ask`, {
      method: "POST", body: jsonBody({ question }),
    });
    state.activeSession = await api(`/sessions/${state.activeSession.conversation_id}`);
    await loadCommon();
  } catch (error) {
    toast(error.message, true);
    state.activeSession = await api(`/sessions/${state.activeSession.conversation_id}`);
  } finally {
    state.sending = false;
    renderShell();
  }
}

async function loadLibrary() {
  [state.papers, state.jobs] = await Promise.all([api("/papers"), api("/ingest/jobs")]);
}

function renderLibrary() {
  return `<section class="page"><div class="content-width">
    <div class="page-head"><div><h2>我的论文库</h2><p>每篇 PDF 都会经过 MinerU 解析，再进入你的私有检索索引。</p></div><button class="secondary" id="refresh-library">刷新状态</button></div>
    <label class="upload-drop" id="upload-drop"><input type="file" id="paper-file" accept="application/pdf,.pdf" /><div><strong>拖放 PDF 到这里，或点击选择</strong><span>最大 ${stateMaxMb()} MB，最多 ${stateMaxPages()} 页</span></div></label>
    <form class="arxiv-box" id="arxiv-search"><input name="query" maxlength="1000" placeholder="搜索 arXiv，例如：agentic RAG citation verification" required /><button class="secondary" type="submit">搜索候选</button></form>
    ${renderCandidates()}
    <div class="toolbar"><input class="search" id="paper-search" placeholder="按标题或文件名筛选" /><select id="paper-status"><option value="">全部状态</option><option value="ready">可检索</option><option value="queued">等待处理</option><option value="parsing">正在解析</option><option value="indexing">正在索引</option><option value="failed">失败</option></select></div>
    <div class="paper-table" id="paper-table">${paperRows(state.papers)}</div>
  </div></section>`;
}

function stateMaxMb() { return 30; }
function stateMaxPages() { return 300; }

function renderCandidates() {
  if (!state.candidates.length) return "";
  return `<div class="candidate-list">${state.candidates.map((candidate) => `<article class="candidate"><div><h4>${escapeHtml(candidate.title)}</h4><p>${escapeHtml(candidate.summary)}</p><div class="activity-meta">${escapeHtml(candidate.arxiv_id)} · ${escapeHtml((candidate.authors || []).slice(0,3).join("、"))}</div></div><button class="primary" data-confirm-arxiv="${escapeHtml(candidate.arxiv_id)}">确认入库</button></article>`).join("")}</div>`;
}

function paperRows(papers) {
  if (!papers.length) return `<div class="empty-state">论文库还是空的。上传 PDF 或从 arXiv 添加第一篇论文。</div>`;
  return `<div class="paper-row header"><div>论文</div><div>来源</div><div>状态</div><div></div></div>${papers.map((paper) => `<div class="paper-row"><div><div class="paper-title">${escapeHtml(paper.title)}</div><div class="paper-file">${escapeHtml(paper.original_filename)}${paper.error ? ` · ${escapeHtml(paper.error)}` : ""}</div></div><div class="activity-meta">${paper.origin === "arxiv" ? "arXiv" : "本地上传"}</div><div><span class="status ${paper.status}">${statusLabel(paper.status)}</span></div><div class="paper-actions">${paper.status === "ready" ? `<button class="ghost" data-preview-paper="${paper.paper_id}">预览</button>` : ""}${paper.status === "failed" ? `<button class="ghost" data-retry-paper="${paper.paper_id}">重试</button>` : ""}<button class="ghost" data-delete-paper="${paper.paper_id}" ${["queued","parsing","indexing","deleting"].includes(paper.status) ? "disabled" : ""}>删除</button></div></div>`).join("")}`;
}

function bindLibrary() {
  document.querySelector("#refresh-library")?.addEventListener("click", async () => { await loadLibrary(); renderShell(); });
  const fileInput = document.querySelector("#paper-file");
  fileInput?.addEventListener("change", () => fileInput.files[0] && uploadPaper(fileInput.files[0]));
  const drop = document.querySelector("#upload-drop");
  ["dragenter", "dragover"].forEach((name) => drop?.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => drop?.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop?.addEventListener("drop", (event) => event.dataTransfer.files[0] && uploadPaper(event.dataTransfer.files[0]));
  document.querySelector("#arxiv-search")?.addEventListener("submit", searchArxiv);
  document.querySelectorAll("[data-confirm-arxiv]").forEach((button) => button.addEventListener("click", () => confirmArxiv(button.dataset.confirmArxiv)));
  document.querySelector("#paper-search")?.addEventListener("input", filterPapers);
  document.querySelector("#paper-status")?.addEventListener("change", filterPapers);
  bindPaperActions();
}

function bindPaperActions() {
  document.querySelectorAll("[data-preview-paper]").forEach((button) => button.addEventListener("click", () => {
    const paper = state.papers.find((item) => item.paper_id === button.dataset.previewPaper);
    openViewer({ source_kind: "library_pdf", preview_kind: "pdf", paper_id: paper.paper_id, paper_title: paper.title, section: "全文", page_start: 1 });
  }));
  document.querySelectorAll("[data-retry-paper]").forEach((button) => button.addEventListener("click", async () => {
    try { await api(`/papers/${button.dataset.retryPaper}/retry`, { method: "POST" }); toast("已重新加入解析队列"); await loadLibrary(); renderShell(); } catch (error) { toast(error.message, true); }
  }));
  document.querySelectorAll("[data-delete-paper]").forEach((button) => button.addEventListener("click", async () => {
    const paper = state.papers.find((item) => item.paper_id === button.dataset.deletePaper);
    if (!await confirmModal("永久删除这篇论文？", `《${paper.title}》的 PDF、sidecar 与索引内容都会被移除。`)) return;
    try { await api(`/papers/${paper.paper_id}`, { method: "DELETE" }); toast("论文已删除"); await loadLibrary(); renderShell(); } catch (error) { toast(error.message, true); }
  }));
}

function filterPapers() {
  const query = document.querySelector("#paper-search").value.trim().toLowerCase();
  const status = document.querySelector("#paper-status").value;
  const filtered = state.papers.filter((paper) => (!status || paper.status === status) && (!query || `${paper.title} ${paper.original_filename}`.toLowerCase().includes(query)));
  document.querySelector("#paper-table").innerHTML = paperRows(filtered);
  bindPaperActions();
}

async function uploadPaper(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) { toast("只支持 PDF 文件", true); return; }
  const form = new FormData(); form.append("file", file);
  try { toast("正在上传并校验 PDF…"); await api("/papers/upload", { method: "POST", body: form }); toast("已加入解析队列"); await loadLibrary(); renderShell(); } catch (error) { toast(error.message, true); }
}

async function searchArxiv(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  try {
    const result = await api("/arxiv/ingest/proposals", { method: "POST", body: jsonBody({ query: new FormData(event.currentTarget).get("query"), max_results: 5 }) });
    state.candidates = result.candidates; state.proposalId = result.proposal_id; renderShell();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

async function confirmArxiv(arxivId) {
  try { await api("/arxiv/ingest/confirm", { method: "POST", body: jsonBody({ proposal_id: state.proposalId, arxiv_id: arxivId }) }); state.candidates = []; state.proposalId = null; toast("已确认，论文正在后台下载解析"); await loadLibrary(); renderShell(); } catch (error) { toast(error.message, true); }
}

function renderSettings() {
  const editing = state.profiles.find((item) => item.profile_id === state.editProfileId);
  return `<section class="page"><div class="content-width"><div class="page-head"><div><h2>账户与模型</h2><p>管理本地账户，以及每次问答使用的 OpenAI-compatible 模型。</p></div></div>
    <div class="settings-grid">
      <div>
        <section class="panel settings-section"><h3>账户资料</h3><p class="section-note">账户只存在于这台设备。密码不会以明文保存。</p><form id="account-form"><div class="field"><label>用户名</label><input value="${escapeHtml(state.auth.user.username)}" disabled /></div><div class="field"><label>显示名称</label><input name="display_name" maxlength="80" value="${escapeHtml(state.auth.user.display_name)}" required /></div><button class="secondary" type="submit">保存名称</button></form></section>
        <section class="panel settings-section" style="margin-top:16px"><h3>修改密码</h3><p class="section-note">修改后，其他浏览器上的登录会话会被撤销。</p><form id="password-form"><div class="field"><label>当前密码</label><input name="current_password" type="password" required /></div><div class="field"><label>新密码</label><input name="new_password" type="password" minlength="10" required /></div><button class="secondary" type="submit">更新密码</button></form></section>
        <section class="panel settings-section" style="margin-top:16px"><h3>登录会话</h3><div id="account-sessions"><div class="thinking">正在读取</div></div><button class="danger-button" id="logout-all" style="margin-top:14px">退出所有会话</button></section>
      </div>
      <div>
        <section class="panel settings-section"><h3>${editing ? "编辑模型配置" : "添加模型配置"}</h3><p class="section-note">API key 使用 AES-GCM 加密保存；测试连接仅在你点击时调用模型。</p>
          <form id="profile-form">
            <div class="two-fields"><div class="field"><label>配置名称</label><input name="name" maxlength="80" value="${escapeHtml(editing?.name || "")}" placeholder="例如：DeepSeek 日常" required /></div><div class="field"><label>供应商</label><input name="provider" maxlength="40" value="${escapeHtml(editing?.provider || "openai-compatible")}" required /></div></div>
            <div class="field"><label>API Base</label><input name="api_base" maxlength="500" value="${escapeHtml(editing?.api_base || "https://api.openai.com/v1")}" required /><div class="field-help">公网地址必须使用 HTTPS；本地地址需在 config.yaml 中显式允许。</div></div>
            <div class="field"><label>模型名称</label><input name="model_name" maxlength="160" value="${escapeHtml(editing?.model_name || "")}" placeholder="例如：gpt-4.1-mini" required /></div>
            <div class="field"><label>API key ${editing ? "（留空表示不修改）" : ""}</label><input name="api_key" type="password" maxlength="1000" ${editing ? "" : "required"} placeholder="sk-…" autocomplete="new-password" /></div>
            <label style="display:flex;gap:8px;align-items:center;margin-bottom:16px;font-size:12px"><input name="is_default" type="checkbox" ${editing?.is_default ? "checked" : ""} />设为默认配置</label>
            <div style="display:flex;gap:8px"><button class="primary" type="submit">${editing ? "保存修改" : "添加配置"}</button>${editing ? `<button class="secondary" type="button" id="cancel-edit">取消</button>` : ""}</div>
          </form>
        </section>
        <section class="panel settings-section" style="margin-top:16px"><h3>已保存的配置</h3><p class="section-note">密钥保存后仅显示末四位，无法从页面取回。</p><div>${state.profiles.length ? state.profiles.map(profileCard).join("") : `<div class="empty-state">还没有模型配置，对话功能暂不可用。</div>`}</div></section>
      </div>
    </div>
  </div></section>`;
}

function profileCard(profile) {
  return `<div class="profile-card"><div class="profile-main"><strong>${escapeHtml(profile.name)} ${profile.is_default ? `<span class="default-tag">默认</span>` : ""}</strong><span>${escapeHtml(profile.model_name)} · key …${escapeHtml(profile.key_last4)} · ${escapeHtml(profile.api_base)}</span></div><button class="ghost" data-test-profile="${profile.profile_id}">测试</button><button class="ghost" data-edit-profile="${profile.profile_id}">编辑</button><button class="ghost" data-delete-profile="${profile.profile_id}">删除</button></div>`;
}

function bindSettings() {
  loadAccountSessions();
  document.querySelector("#account-form")?.addEventListener("submit", async (event) => {
    event.preventDefault(); try { const result = await api("/account", { method: "PATCH", body: jsonBody(Object.fromEntries(new FormData(event.currentTarget))) }); setAuthenticated(result); toast("显示名称已更新"); renderShell(); } catch (error) { toast(error.message, true); }
  });
  document.querySelector("#password-form")?.addEventListener("submit", async (event) => {
    event.preventDefault(); try { await api("/account/change-password", { method: "POST", body: jsonBody(Object.fromEntries(new FormData(event.currentTarget))) }); event.currentTarget.reset(); toast("密码已更新，其他会话已退出"); } catch (error) { toast(error.message, true); }
  });
  document.querySelector("#logout-all")?.addEventListener("click", async () => { if (await confirmModal("退出所有会话？", "包括当前浏览器在内的所有登录都会失效。")) { await api("/account/logout-all", { method: "POST" }); renderAuth(); } });
  document.querySelector("#profile-form")?.addEventListener("submit", saveProfile);
  document.querySelector("#cancel-edit")?.addEventListener("click", () => { state.editProfileId = null; renderShell(); });
  document.querySelectorAll("[data-edit-profile]").forEach((button) => button.addEventListener("click", () => { state.editProfileId = button.dataset.editProfile; renderShell(); }));
  document.querySelectorAll("[data-test-profile]").forEach((button) => button.addEventListener("click", async () => { button.disabled = true; try { const result = await api(`/model-profiles/${button.dataset.testProfile}/test`, { method: "POST" }); toast(`连接成功，耗时 ${result.duration_ms} ms`); } catch (error) { toast(error.message, true); } finally { button.disabled = false; } }));
  document.querySelectorAll("[data-delete-profile]").forEach((button) => button.addEventListener("click", async () => { if (!await confirmModal("删除模型配置？", "使用该配置的对话需要重新选择模型后才能继续。")) return; try { await api(`/model-profiles/${button.dataset.deleteProfile}`, { method: "DELETE" }); await loadCommon(); renderShell(); } catch (error) { toast(error.message, true); } }));
}

async function saveProfile(event) {
  event.preventDefault(); const data = Object.fromEntries(new FormData(event.currentTarget)); data.is_default = new FormData(event.currentTarget).has("is_default"); if (!data.api_key) delete data.api_key;
  try { await api(state.editProfileId ? `/model-profiles/${state.editProfileId}` : "/model-profiles", { method: state.editProfileId ? "PATCH" : "POST", body: jsonBody(data) }); state.editProfileId = null; await loadCommon(); toast("模型配置已保存"); renderShell(); } catch (error) { toast(error.message, true); }
}

async function loadAccountSessions() {
  const target = document.querySelector("#account-sessions"); if (!target) return;
  try { const sessions = await api("/account/sessions"); target.innerHTML = sessions.map((item) => `<div class="session-card"><strong>${item.current ? "当前会话" : "其他会话"}</strong><div class="activity-meta">${escapeHtml(item.user_agent || "未知浏览器")} · ${formatDate(item.last_seen_at)}</div></div>`).join(""); } catch (error) { target.textContent = error.message; }
}

async function logout() {
  try { await api("/auth/logout", { method: "POST" }); } catch (_) {}
  setCsrfToken(""); state.auth = null; state.activeSession = null; location.hash = ""; renderAuth();
}

function citationClick(event) {
  const citation = event.target.closest("[data-citation-key]");
  if (!citation) return;
  if (citation.tagName === "A" && (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button === 1)) return;
  event.preventDefault();
  const source = sourceForKey(citation.dataset.citationKey);
  if (source) openViewer(source);
}

function openViewer(source) {
  state.viewer = source; state.viewerTab = "preview"; renderViewer();
}
function closeViewer() { state.viewer = null; renderViewer(); }

function renderViewer() {
  const viewer = document.querySelector("#viewer"); const backdrop = document.querySelector("#viewer-backdrop"); if (!viewer || !backdrop) return;
  viewer.classList.toggle("open", Boolean(state.viewer)); backdrop.classList.toggle("open", Boolean(state.viewer));
  if (!state.viewer) { viewer.replaceChildren(); return; }
  const source = state.viewer; const external = source.source_kind === "external_url";
  const page = Math.max(1, Number(source.page_start) || 1); const url = external ? source.citation_url : `/api/papers/${encodeURIComponent(source.paper_id)}/pdf?preview=${++previewNonce}#page=${page}`;
  viewer.innerHTML = `<div class="viewer-head"><div class="viewer-title"><strong>${escapeHtml(source.paper_title || "来源预览")}</strong><span>${escapeHtml(source.section || (external ? "外部网页" : `第 ${page} 页`))}</span></div>${external ? `<a class="secondary" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">浏览器打开</a>` : `<a class="secondary" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">新窗口</a>`}<button class="ghost" id="close-viewer">关闭</button></div>${external ? `<div class="viewer-tabs"><button class="viewer-tab ${state.viewerTab === "preview" ? "active" : ""}" data-viewer-tab="preview">网页预览</button><button class="viewer-tab ${state.viewerTab === "summary" ? "active" : ""}" data-viewer-tab="summary">来源摘要</button></div>` : ""}${state.viewerTab === "summary" && external ? `<div class="viewer-summary"><div class="eyebrow">EXTERNAL SOURCE</div><h3>${escapeHtml(source.paper_title)}</h3><p>${escapeHtml(source.snippet || "该来源没有返回摘要。")}</p><a class="primary" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">在浏览器中打开论文</a></div>` : `<iframe class="viewer-frame" src="${escapeHtml(url)}" title="来源预览" ${external ? `sandbox="allow-scripts allow-same-origin allow-forms allow-popups" referrerpolicy="no-referrer"` : ""}></iframe>`}`;
  document.querySelector("#close-viewer")?.addEventListener("click", closeViewer);
  document.querySelectorAll("[data-viewer-tab]").forEach((button) => button.addEventListener("click", () => { state.viewerTab = button.dataset.viewerTab; renderViewer(); }));
}

function startPolling() {
  if (!state.jobs.some((job) => ["queued", "parsing", "indexing"].includes(job.status))) return;
  pollTimer = window.setInterval(async () => {
    try { await loadLibrary(); if (state.route === "library") renderShell(); if (!state.jobs.some((job) => ["queued", "parsing", "indexing"].includes(job.status))) stopPolling(); } catch (_) {}
  }, 3500);
}
function stopPolling() { if (pollTimer) window.clearInterval(pollTimer); pollTimer = null; }

function toast(message, error = false) {
  const root = document.querySelector("#toast-root"); const item = document.createElement("div"); item.className = `toast ${error ? "error" : ""}`; item.textContent = message; root.append(item); window.setTimeout(() => item.remove(), 3800);
}

function confirmModal(title, message) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div"); backdrop.className = "modal-backdrop"; backdrop.innerHTML = `<div class="modal" role="dialog" aria-modal="true"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p><div class="modal-actions"><button class="secondary" data-cancel>取消</button><button class="danger-button" data-confirm>确认</button></div></div>`; document.body.append(backdrop);
    const finish = (value) => { backdrop.remove(); resolve(value); };
    backdrop.querySelector("[data-cancel]").addEventListener("click", () => finish(false)); backdrop.querySelector("[data-confirm]").addEventListener("click", () => finish(true)); backdrop.addEventListener("click", (event) => { if (event.target === backdrop) finish(false); });
  });
}

function maybeOfferLegacyImport() {
  if (localStorage.getItem("paperdesk_legacy_prompted")) return;
  let legacy = [];
  try { legacy = JSON.parse(localStorage.getItem("rag_sessions") || "[]"); } catch (_) {}
  if (!Array.isArray(legacy) || !legacy.length) { localStorage.setItem("paperdesk_legacy_prompted", "1"); return; }
  confirmModal("导入旧版浏览器对话？", `检测到 ${legacy.length} 个旧对话。只有在服务端保存成功后，旧数据才会被清理。`).then(async (accepted) => {
    localStorage.setItem("paperdesk_legacy_prompted", "1"); if (!accepted) return;
    try {
      const sessions = legacy.slice(0, 50).map((item) => ({
        title: item.title || "旧对话",
        messages: (item.messages || [])
          .filter((message) => ["user", "assistant"].includes(message.role))
          .slice(0, 100)
          .map((message) => ({
            role: message.role,
            content: String(message.content || "").slice(0, 20000),
          })),
      }));
      await api("/sessions/import", {
        method: "POST",
        body: jsonBody({ sessions }),
      });
      localStorage.removeItem("rag_sessions");
      localStorage.removeItem("rag_active");
      toast("旧对话已导入");
      await loadCommon();
    } catch (error) {
      toast(`导入失败：${error.message}`, true);
    }
  });
}
