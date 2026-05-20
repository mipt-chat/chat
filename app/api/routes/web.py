# ruff: noqa: E501
"""Minimal web chat client served by the backend."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["web"])


_CHAT_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CustomerSupportBot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef3f8;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e1ea;
      --accent: #0f766e;
      --accent-2: #1d4ed8;
      --danger: #b42318;
      --shadow: 0 10px 32px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        linear-gradient(180deg, rgba(15, 118, 110, 0.10), rgba(246, 247, 249, 0) 260px),
        var(--bg);
    }
    .shell {
      width: min(1180px, calc(100vw - 32px));
      min-height: 100vh;
      margin: 0 auto;
      padding: 24px 0;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 18px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.12);
    }
    .layout {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
    }
    .chat,
    .sources {
      min-height: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .chat {
      display: grid;
      grid-template-rows: 1fr auto;
      overflow: hidden;
    }
    .messages {
      min-height: 0;
      overflow: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .message {
      max-width: min(720px, 88%);
      padding: 11px 13px;
      border-radius: 8px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      align-self: flex-end;
      color: white;
      background: var(--accent-2);
    }
    .message.assistant {
      align-self: flex-start;
      background: var(--surface-2);
    }
    .message.error {
      align-self: flex-start;
      color: var(--danger);
      background: #fff1f0;
      border: 1px solid #fecdca;
    }
    form {
      border-top: 1px solid var(--line);
      padding: 14px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      background: #fbfcfd;
    }
    textarea {
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      color: var(--text);
      background: white;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }
    button {
      min-width: 112px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      font: inherit;
      font-weight: 650;
      color: white;
      background: var(--accent);
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .sources {
      padding: 18px;
      overflow: auto;
    }
    .sources h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .source {
      padding: 10px 0;
      border-top: 1px solid var(--line);
    }
    .source:first-of-type { border-top: 0; }
    .source strong {
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .source span {
      color: var(--muted);
      font-size: 12px;
    }
    .source p {
      margin: 6px 0 0;
      color: #344054;
      font-size: 13px;
      display: -webkit-box;
      -webkit-line-clamp: 4;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    @media (max-width: 860px) {
      .shell {
        width: min(100vw - 20px, 720px);
        padding: 12px 0;
      }
      header { align-items: flex-start; }
      .layout { grid-template-columns: 1fr; }
      .sources { max-height: 240px; }
      form { grid-template-columns: 1fr; }
      button { min-height: 44px; }
      .message { max-width: 96%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>CustomerSupportBot</h1>
      <div class="status"><span class="dot"></span><span id="status">Готов к вопросам</span></div>
    </header>
    <main class="layout">
      <section class="chat" aria-label="Чат поддержки">
        <div id="messages" class="messages"></div>
        <form id="form">
          <textarea
            id="message"
            name="message"
            maxlength="5000"
            placeholder="Введите вопрос по базе знаний"
            required
          ></textarea>
          <button id="send" type="submit">Отправить</button>
        </form>
      </section>
      <aside class="sources" aria-label="Источники ответа">
        <h2>Источники</h2>
        <div id="sources">Появятся после ответа.</div>
      </aside>
    </main>
  </div>
  <script>
    const messages = document.querySelector("#messages");
    const sources = document.querySelector("#sources");
    const form = document.querySelector("#form");
    const input = document.querySelector("#message");
    const send = document.querySelector("#send");
    const status = document.querySelector("#status");
    let sessionId = localStorage.getItem("support_chat_session_id");

    function appendMessage(role, text) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.textContent = text;
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
      return node;
    }

    function renderSources(items) {
      if (!items || !items.length) {
        sources.textContent = "Релевантные источники не найдены.";
        return;
      }
      sources.replaceChildren(...items.slice(0, 5).map((item) => {
        const node = document.createElement("div");
        node.className = "source";
        const title = document.createElement("strong");
        title.textContent = item.source_path || item.chunk_id;
        const score = document.createElement("span");
        score.textContent = `score ${Number(item.score || 0).toFixed(3)}`;
        const text = document.createElement("p");
        text.textContent = item.text || "";
        node.append(title, score, text);
        return node;
      }));
    }

    async function readSse(response, handlers) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let eventName = "message";
      let dataLines = [];

      function flush() {
        if (!dataLines.length) return;
        const raw = dataLines.join("\\n");
        const payload = JSON.parse(raw);
        if (handlers[eventName]) handlers[eventName](payload);
        eventName = "message";
        dataLines = [];
      }

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\\r?\\n/);
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line === "") {
            flush();
          } else if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        }
      }
      if (buffer) {
        for (const line of buffer.split(/\\r?\\n/)) {
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        }
      }
      flush();
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      appendMessage("user", text);
      input.value = "";
      input.focus();
      send.disabled = true;
      status.textContent = "Генерируется ответ";
      sources.textContent = "Идёт поиск по базе знаний.";
      const assistant = appendMessage("assistant", "");

      try {
        const response = await fetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        });
        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }
        await readSse(response, {
          session(payload) {
            sessionId = payload.session_id;
            localStorage.setItem("support_chat_session_id", sessionId);
          },
          token(payload) {
            assistant.textContent += payload.text;
            messages.scrollTop = messages.scrollHeight;
          },
          sources(payload) {
            renderSources(payload.sources);
          },
          done(payload) {
            assistant.textContent = payload.answer || assistant.textContent;
            renderSources(payload.sources);
          },
          error(payload) {
            assistant.className = "message error";
            assistant.textContent = payload.detail || payload.error;
            sources.textContent = "Не удалось получить ответ.";
          },
        });
      } catch (error) {
        assistant.className = "message error";
        assistant.textContent = String(error);
        sources.textContent = "Не удалось получить ответ.";
      } finally {
        send.disabled = false;
        status.textContent = "Готов к вопросам";
      }
    });
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def web_chat() -> HTMLResponse:
    """Serve the built-in web chat client."""

    return HTMLResponse(_CHAT_HTML)
