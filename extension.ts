import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

type WebviewRequestMessage = {
  kind: 'request';
  id: number;
  method: string;
  params?: any;
};

type WebviewResponseMessage = {
  kind: 'response';
  id: number;
  ok: boolean;
  result?: any;
  error?: string;
};

type GenerateOptions = {
  prompt: string;
  engine?: string;
  seconds?: number;
  outputPath?: string;
  post?: boolean;
  edit?: boolean;
};

type ExportPackOptions = {
  manifestPath: string;
  zipPath: string;
};

type OpenUiOptions = {
  wavPath?: string;
};

type OpenWebUiOptions = {
  host?: string;
  port?: number;
  mode?: 'control-panel' | 'legacy';
};

let webUiProc: cp.ChildProcessWithoutNullStreams | null = null;
let webUiUrl: string | null = null;
let webUiPanel: vscode.WebviewPanel | null = null;

function slugifyFileStem(input: string): string {
  const s = String(input || '').trim().toLowerCase();
  if (!s) return 'sound';
  const cleaned = s
    .replace(/[^a-z0-9\s._-]+/g, '')
    .replace(/[\s._-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return cleaned.slice(0, 60) || 'sound';
}

function formatBackendError(e: any): string {
  const msg = String(e?.message || e || 'Unknown error');
  const stderr = (e && typeof e.stderr === 'string') ? e.stderr.trim() : '';
  if (!stderr) return msg;
  return `${msg}\n\n${stderr}`;
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let value = '';
  for (let i = 0; i < 32; i++) value += chars.charAt(Math.floor(Math.random() * chars.length));
  return value;
}

function parseJsonLineBestEffort(text: string): any | null {
  const lines = String(text || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  // Heuristic: backend may log + then print a JSON line.
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line.startsWith('{') || !line.endsWith('}')) continue;
    try {
      return JSON.parse(line);
    } catch {
      // keep trying
    }
  }
  return null;
}

function resolvePythonCommand(repoRoot: string): string {
  const cfgPython = String(vscode.workspace.getConfiguration('sondbound').get('pythonPath') || '').trim();
  if (cfgPython) return cfgPython;

  const envPython = String(process.env.SOUNDGEN_PYTHON || '').trim();
  if (envPython) return envPython;

  const venvWin = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  const venvPosix = path.join(repoRoot, '.venv', 'bin', 'python');
  if (process.platform === 'win32' && fs.existsSync(venvWin)) return venvWin;
  if (process.platform !== 'win32' && fs.existsSync(venvPosix)) return venvPosix;
  return 'python';
}

function defaultEngineFromConfig(): string {
  return String(vscode.workspace.getConfiguration('sondbound').get('defaultEngine') || 'rfxgen');
}

function defaultSecondsFromConfig(): number {
  const n = Number(vscode.workspace.getConfiguration('sondbound').get('defaultSeconds') ?? 3.0);
  return Number.isFinite(n) && n > 0 ? n : 3.0;
}

function defaultPostFromConfig(): boolean {
  return Boolean(vscode.workspace.getConfiguration('sondbound').get('defaultPost') ?? true);
}

function defaultOutputSubdirFromConfig(): string {
  const s = String(vscode.workspace.getConfiguration('sondbound').get('defaultOutputSubdir') || 'outputs').trim();
  return s || 'outputs';
}

function defaultWebUiHostFromConfig(): string {
  const s = String(vscode.workspace.getConfiguration('sondbound').get('webUiHost') || '127.0.0.1').trim();
  return s || '127.0.0.1';
}

function defaultWebUiPortFromConfig(): number {
  const n = Number(vscode.workspace.getConfiguration('sondbound').get('webUiPort') ?? 7860);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 7860;
}

function defaultWebUiModeFromConfig(): 'control-panel' | 'legacy' {
  const s = String(vscode.workspace.getConfiguration('sondbound').get('webUiMode') || 'control-panel').trim().toLowerCase();
  return s === 'legacy' ? 'legacy' : 'control-panel';
}

function makeWebUiHtml(webview: vscode.Webview, url: string): string {
  const csp = [
    "default-src 'none'",
    // Allow iframing local Gradio server
    "frame-src http://127.0.0.1:* http://localhost:*",
    "child-src http://127.0.0.1:* http://localhost:*",
    // Minimal styles for layout
    `style-src ${webview.cspSource} 'unsafe-inline'`,
  ].join('; ');

  const safeUrl = String(url);

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>SÖNDBÖUND Web UI</title>
  <style>
    html, body { height: 100%; padding: 0; margin: 0; }
    .wrap { height: 100%; display: flex; flex-direction: column; }
    .bar { padding: 6px 10px; font-size: 12px; border-bottom: 1px solid rgba(127,127,127,0.25); }
    .bar a { color: inherit; }
    iframe { flex: 1; width: 100%; border: 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">Local URL: <a href="${safeUrl}">${safeUrl}</a></div>
    <iframe
      src="${safeUrl}"
      sandbox="allow-same-origin allow-scripts allow-forms allow-downloads allow-popups"
      allow="clipboard-read; clipboard-write"
    ></iframe>
  </div>
</body>
</html>`;
}

async function startWebUiServer(
  repoRoot: string,
  storageDir: string,
  output: vscode.OutputChannel,
  options?: OpenWebUiOptions
): Promise<string> {
  if (webUiProc && webUiUrl) return webUiUrl;

  const host = String(options?.host || defaultWebUiHostFromConfig()).trim() || '127.0.0.1';
  const port = options?.port != null ? Number(options.port) : defaultWebUiPortFromConfig();
  const mode = options?.mode || defaultWebUiModeFromConfig();
  if (!Number.isFinite(port) || port <= 0) {
    throw new Error('web UI port must be a positive number');
  }

  const py = resolvePythonCommand(repoRoot);
  const fullArgs = ['-m', 'soundgen.web'];

  const env = { ...process.env } as Record<string, string>;
  env.SOUNDGEN_DATA_DIR = storageDir;
  env.GRADIO_ANALYTICS_ENABLED = 'False';
  env.PYTHONPATH = prependEnvPath(env.PYTHONPATH, path.join(repoRoot, 'src'));

  // Gradio respects these environment variables.
  env.GRADIO_SERVER_NAME = host;
  env.GRADIO_SERVER_PORT = String(Math.floor(port));
  if (mode === 'legacy') env.SOUNDGEN_WEB_UI = 'legacy';

  output.appendLine(`[webui] starting: ${py} ${fullArgs.join(' ')}`);
  output.appendLine(`[webui] host=${host} port=${Math.floor(port)} mode=${mode}`);

  const child = cp.spawn(py, fullArgs, {
    cwd: repoRoot,
    env,
    windowsHide: true,
  });

  webUiProc = child;
  webUiUrl = null;

  const reset = () => {
    if (webUiProc === child) {
      webUiProc = null;
      webUiUrl = null;
    }
  };

  child.on('exit', (code) => {
    output.appendLine(`[webui] exited with code ${code}`);
    reset();
  });
  child.on('error', (e) => {
    output.appendLine(`[webui] spawn error: ${String((e as any)?.message || e)}`);
    reset();
  });

  const urlRegex = /(https?:\/\/(?:127\.0\.0\.1|localhost|0\.0\.0\.0)(?::\d+)?\/?)/i;
  let stdout = '';
  let stderr = '';
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');

  const maybeCaptureUrl = (chunk: string) => {
    const m = urlRegex.exec(chunk);
    if (!m) return;
    const raw = m[1];
    // Normalize 0.0.0.0 to a real local host the Webview can reach.
    const normalized = raw.replace('0.0.0.0', host === '0.0.0.0' ? '127.0.0.1' : host);
    webUiUrl = normalized;
  };

  child.stdout.on('data', (d) => {
    const s = String(d);
    stdout += s;
    output.append(s);
    maybeCaptureUrl(s);
  });
  child.stderr.on('data', (d) => {
    const s = String(d);
    stderr += s;
    output.append(s);
    maybeCaptureUrl(s);
  });

  const timeoutMs = 30000;
  const started = await new Promise<string>((resolve, reject) => {
    const start = Date.now();
    const timer = setInterval(() => {
      if (webUiUrl) {
        clearInterval(timer);
        return resolve(webUiUrl);
      }
      if (!webUiProc) {
        clearInterval(timer);
        const hint = (stderr || stdout || '').trim();
        return reject(new Error(hint || 'Web UI process exited before it printed a URL.'));
      }
      if (Date.now() - start > timeoutMs) {
        clearInterval(timer);
        return reject(new Error('Timed out waiting for the web UI to start.'));
      }
    }, 200);
  });

  return started;
}

function prependEnvPath(existing: string | undefined, toPrepend: string): string {
  const delimiter = path.delimiter;
  const parts = [toPrepend, ...(existing ? [existing] : [])].filter(Boolean);
  // De-dupe in a simple way.
  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of parts.join(delimiter).split(delimiter)) {
    const norm = p.trim();
    if (!norm) continue;
    if (seen.has(norm)) continue;
    seen.add(norm);
    out.push(norm);
  }
  return out.join(delimiter);
}

async function runBackendOnce(
  repoRoot: string,
  storageDir: string,
  args: string[],
  opts: { timeoutMs?: number } = {}
): Promise<{ stdout: string; stderr: string }>
{
  const py = resolvePythonCommand(repoRoot);
  const fullArgs = ['-m', 'soundgen.app', ...args];

  const env = { ...process.env } as Record<string, string>;
  env.SOUNDGEN_DATA_DIR = storageDir;
  env.GRADIO_ANALYTICS_ENABLED = 'False';
  // Make src-layout importable without requiring editable install.
  env.PYTHONPATH = prependEnvPath(env.PYTHONPATH, path.join(repoRoot, 'src'));

  const timeoutMs = opts.timeoutMs ?? 0;

  return await new Promise((resolve, reject) => {
    const child = cp.spawn(py, fullArgs, {
      cwd: repoRoot,
      env,
      windowsHide: true
    });

    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (d) => { stdout += String(d); });
    child.stderr.on('data', (d) => { stderr += String(d); });

    let timer: NodeJS.Timeout | null = null;
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        try { child.kill(); } catch {}
        reject(new Error('Timed out running backend command.'));
      }, timeoutMs);
    }

    child.on('error', (e) => {
      if (timer) clearTimeout(timer);
      reject(e);
    });

    child.on('exit', (code) => {
      if (timer) clearTimeout(timer);
      if (code === 0) return resolve({ stdout, stderr });
      const err = new Error(`Backend command failed (exit ${code}).`);
      (err as any).stdout = stdout;
      (err as any).stderr = stderr;
      reject(err);
    });
  });
}

async function runPythonModuleOnce(
  repoRoot: string,
  storageDir: string,
  moduleName: string,
  args: string[],
  opts: { timeoutMs?: number } = {}
): Promise<{ stdout: string; stderr: string }>
{
  const py = resolvePythonCommand(repoRoot);
  const fullArgs = ['-m', moduleName, ...args];

  const env = { ...process.env } as Record<string, string>;
  env.SOUNDGEN_DATA_DIR = storageDir;
  env.GRADIO_ANALYTICS_ENABLED = 'False';
  env.PYTHONPATH = prependEnvPath(env.PYTHONPATH, path.join(repoRoot, 'src'));

  const timeoutMs = opts.timeoutMs ?? 0;

  return await new Promise((resolve, reject) => {
    const child = cp.spawn(py, fullArgs, {
      cwd: repoRoot,
      env,
      windowsHide: true
    });

    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (d) => { stdout += String(d); });
    child.stderr.on('data', (d) => { stderr += String(d); });

    let timer: NodeJS.Timeout | null = null;
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        try { child.kill(); } catch {}
        reject(new Error('Timed out running backend command.'));
      }, timeoutMs);
    }

    child.on('error', (e) => {
      if (timer) clearTimeout(timer);
      reject(e);
    });

    child.on('exit', (code) => {
      if (timer) clearTimeout(timer);
      if (code === 0) return resolve({ stdout, stderr });
      const err = new Error(`Backend command failed (exit ${code}).`);
      (err as any).stdout = stdout;
      (err as any).stderr = stderr;
      reject(err);
    });
  });
}

function pluginhostExePath(repoRoot: string): string | null {
  const candidates = [
    path.join(repoRoot, 'native', 'pluginhost', 'build', 'Release', 'soundgen_pluginhost.exe'),
    path.join(repoRoot, 'native', 'pluginhost', 'build', 'soundgen_pluginhost.exe'),
    path.join(repoRoot, 'native', 'pluginhost', 'build', 'Debug', 'soundgen_pluginhost.exe'),
  ];
  for (const c of candidates) {
    try {
      if (c && fs.existsSync(c)) return c;
    } catch {
      // ignore
    }
  }
  return null;
}

async function runPluginhost(repoRoot: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  const exe = pluginhostExePath(repoRoot);
  if (!exe) {
    throw new Error('soundgen_pluginhost.exe not found. Build native/pluginhost first.');
  }

  return await new Promise((resolve, reject) => {
    const child = cp.spawn(exe, args, { windowsHide: true, cwd: path.dirname(exe) });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (d) => { stdout += String(d); });
    child.stderr.on('data', (d) => { stderr += String(d); });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) return resolve({ stdout, stderr });
      reject(new Error((stderr || stdout || '').trim() || `pluginhost exited with code ${code}`));
    });
  });
}

function makeEditorHtml(webview: vscode.Webview, extensionUri: vscode.Uri, initialWavPath?: string): string {
  const nonce = getNonce();

  const bridgeUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'electron', 'editor', 'vscode-bridge.js'));
  const editorJsUri = webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'electron', 'editor', 'editor.js'));

  const csp = [
    "default-src 'none'",
    `img-src ${webview.cspSource} https: data:`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src 'nonce-${nonce}'`,
    `media-src ${webview.cspSource} blob: data:`,
  ].join('; ');

  const initialJson = initialWavPath ? JSON.stringify(String(initialWavPath)) : 'null';

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <title>SÖNDBÖUND Editor</title>
  <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    header { display: flex; gap: 8px; align-items: center; padding: 10px 12px; border-bottom: 1px solid #ddd; }
    header button { padding: 6px 10px; }
    header .spacer { flex: 1; }
    main { display: grid; grid-template-columns: 1fr 320px; height: calc(100vh - 52px); }
    #left { display: flex; flex-direction: column; }
    #waveWrap { flex: 1; background: #fafafa; }
    #wave { width: 100%; height: 100%; display: block; }
    #transport { display: flex; gap: 10px; align-items: center; padding: 10px 12px; border-top: 1px solid #ddd; }
    #right { border-left: 1px solid #ddd; display: flex; flex-direction: column; }
    #fileInfo { padding: 10px 12px; border-bottom: 1px solid #ddd; font-size: 12px; color: #333; }
    #history { flex: 1; overflow: auto; padding: 10px 12px; font-size: 12px; }
    #history .item { padding: 6px 6px; border: 1px solid #eee; border-radius: 6px; margin-bottom: 8px; background: #fff; }
    #history .item .op { font-weight: 600; }
    #status { padding: 10px 12px; border-top: 1px solid #ddd; font-size: 12px; color: #333; white-space: pre-wrap; }
  </style>
</head>
<body>
  <header>
    <button id="openBtn">Open WAV…</button>
    <button id="loadClapBtn" disabled>Load CLAP…</button>
    <button id="loadLv2Btn" disabled>Load LV2…</button>
    <label style="display:flex; align-items:center; gap:6px; font-size:12px;">
      <input id="pluginPreviewChk" type="checkbox" /> Preview plugins
    </label>
    <button id="trimBtn" disabled>Trim</button>
    <button id="cutBtn" disabled>Cut</button>
    <button id="copyBtn" disabled>Copy</button>
    <button id="pasteBtn" disabled>Paste</button>
    <button id="deleteBtn" disabled>Delete</button>
    <button id="silenceBtn" disabled>Insert Silence</button>
    <button id="fadeInBtn" disabled>Fade In</button>
    <button id="fadeOutBtn" disabled>Fade Out</button>
    <button id="normBtn" disabled>Normalize</button>
    <button id="revBtn" disabled>Reverse</button>
    <button id="pitchBtn" disabled>Pitch</button>
    <button id="eqBtn" disabled>EQ (3‑band)</button>
    <div class="spacer"></div>
    <button id="undoBtn" disabled>Undo</button>
    <button id="redoBtn" disabled>Redo</button>
    <button id="exportBtn" disabled>Export WAV…</button>
  </header>

  <main>
    <section id="left">
      <div id="waveWrap"><canvas id="wave"></canvas></div>
      <div id="transport">
        <button id="playBtn" disabled>Play</button>
        <button id="stopBtn" disabled>Stop</button>
        <label><input id="loopChk" type="checkbox" /> Loop selection</label>
        <div id="timeLabel" style="margin-left:auto; font-variant-numeric: tabular-nums;"></div>
      </div>
    </section>

    <aside id="right">
      <div id="fileInfo">No file loaded.</div>
      <div id="history"></div>
      <div id="status"></div>
    </aside>
  </main>

  <script nonce="${nonce}" src="${bridgeUri}"></script>
  <script nonce="${nonce}" src="${editorJsUri}"></script>
  <script nonce="${nonce}">
    (function() {
      const initial = ${initialJson};
      if (!initial) return;
      // Trigger the existing Open button flow (editor.js will call soundgenEditor.openWavDialog).
      window.addEventListener('load', () => {
        try {
          const btn = document.getElementById('openBtn');
          if (btn) btn.click();
        } catch {}
      });
    })();
  </script>
</body>
</html>`;
}

async function openEditorPanel(context: vscode.ExtensionContext, initialWavPath?: string): Promise<void> {
  const panel = vscode.window.createWebviewPanel(
    'sondboundEditor',
    'SÖNDBÖUND Editor',
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [
        vscode.Uri.joinPath(context.extensionUri, 'electron', 'editor'),
      ],
    }
  );

  const repoRoot = context.extensionPath;
  const storageDir = context.globalStorageUri.fsPath;
  fs.mkdirSync(storageDir, { recursive: true });

  const initial = initialWavPath && fs.existsSync(initialWavPath) ? initialWavPath : undefined;
  let forcedOpenWavPath: string | null = initial ? String(initial) : null;

  panel.webview.html = makeEditorHtml(panel.webview, context.extensionUri, initial);

  let lastSessionId: string | null = null;

  panel.onDidDispose(() => {
    if (!lastSessionId) return;
    void runBackendOnce(repoRoot, storageDir, ['editops', 'close', '--session', lastSessionId], { timeoutMs: 120000 })
      .catch(() => { /* ignore */ });
  });

  panel.webview.onDidReceiveMessage(async (raw: any) => {
    const msg = raw as WebviewRequestMessage;
    if (!msg || msg.kind !== 'request' || typeof msg.id !== 'number' || typeof msg.method !== 'string') return;

    const respond = (resp: WebviewResponseMessage) => panel.webview.postMessage(resp);

    try {
      switch (msg.method) {
        case 'openWavDialog': {
          if (forcedOpenWavPath) {
            const p = forcedOpenWavPath;
            forcedOpenWavPath = null;
            respond({ kind: 'response', id: msg.id, ok: true, result: p });
            return;
          }

          const pick = await vscode.window.showOpenDialog({
            title: 'Open WAV',
            canSelectMany: false,
            filters: { WAV: ['wav'] }
          });
          respond({ kind: 'response', id: msg.id, ok: true, result: pick && pick[0] ? pick[0].fsPath : null });
          return;
        }
        case 'saveWavDialog': {
          const uri = await vscode.window.showSaveDialog({
            title: 'Export WAV',
            filters: { WAV: ['wav'] }
          });
          respond({ kind: 'response', id: msg.id, ok: true, result: uri ? uri.fsPath : null });
          return;
        }
        case 'pickClapPluginDialog': {
          const pick = await vscode.window.showOpenDialog({
            title: 'Load CLAP plugin',
            canSelectMany: false,
            filters: { 'CLAP plugin': ['clap'] }
          });
          respond({ kind: 'response', id: msg.id, ok: true, result: pick && pick[0] ? pick[0].fsPath : null });
          return;
        }
        case 'pickLv2BundleDialog': {
          const pick = await vscode.window.showOpenDialog({
            title: 'Load LV2 bundle',
            canSelectMany: false,
            canSelectFiles: false,
            canSelectFolders: true
          });
          respond({ kind: 'response', id: msg.id, ok: true, result: pick && pick[0] ? pick[0].fsPath : null });
          return;
        }
        case 'readFileBase64': {
          const p = String(msg.params?.filePath || '');
          if (!p) throw new Error('readFileBase64 requires filePath');
          const buf = await vscode.workspace.fs.readFile(vscode.Uri.file(p));
          respond({ kind: 'response', id: msg.id, ok: true, result: Buffer.from(buf).toString('base64') });
          return;
        }
        case 'clapListPlugins': {
          const pluginPath = String(msg.params?.pluginPath || '');
          if (!pluginPath) throw new Error('clapListPlugins requires {pluginPath}');
          const res = await runPluginhost(repoRoot, ['clap-list', '--plugin', pluginPath]);
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('clap-list returned invalid JSON');
          if (obj.ok === false) throw new Error(String(obj.error || 'clap-list failed'));
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'clapRenderPreview': {
          const inWav = String(msg.params?.inWav || '');
          const outWav = String(msg.params?.outWav || '');
          const pluginPath = String(msg.params?.pluginPath || '');
          const pluginId = msg.params?.pluginId != null ? String(msg.params.pluginId) : '';
          if (!inWav || !outWav || !pluginPath) {
            throw new Error('clapRenderPreview requires {inWav,outWav,pluginPath}');
          }
          const args: string[] = ['clap-render', '--plugin', pluginPath];
          if (pluginId) args.push('--plugin-id', pluginId);
          args.push('--in', inWav, '--out', outWav);
          await runPluginhost(repoRoot, args);
          respond({ kind: 'response', id: msg.id, ok: true, result: { outWav } });
          return;
        }
        case 'editopsInit': {
          const inPath = String(msg.params?.inPath || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'init', '--in', inPath], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops init returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || lastSessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsInfo': {
          const sessionId = String(msg.params?.sessionId || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'info', '--session', sessionId], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops info returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || sessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsOp': {
          const p = msg.params || {};
          const sessionId = String(p.sessionId || '');
          const type = String(p.type || '');
          const args = ['editops', 'op', '--session', sessionId, '--type', type];
          if (p.start != null && p.end != null) {
            args.push('--start', String(p.start));
            args.push('--end', String(p.end));
          }
          if (p.cursor != null) {
            args.push('--cursor', String(p.cursor));
          }
          if (type === 'silence_insert') {
            args.push('--silence-ms', String(p.silenceMs != null ? p.silenceMs : 250.0));
          }
          if (type === 'fade') {
            args.push('--fade-mode', String(p.fadeMode || 'in'));
            args.push('--fade-ms', String(p.fadeMs != null ? p.fadeMs : 30.0));
          }
          if (type === 'normalize') {
            args.push('--normalize-peak-db', String(p.normalizePeakDb != null ? p.normalizePeakDb : -1.0));
          }
          if (type === 'pitch') {
            args.push('--pitch-semitones', String(p.pitchSemitones != null ? p.pitchSemitones : 0.0));
          }
          if (type === 'eq3') {
            args.push('--eq-low-cut-hz', String(p.eqLowCutHz != null ? p.eqLowCutHz : 80.0));
            args.push('--eq-mid-freq-hz', String(p.eqMidFreqHz != null ? p.eqMidFreqHz : 1200.0));
            args.push('--eq-mid-gain-db', String(p.eqMidGainDb != null ? p.eqMidGainDb : 0.0));
            args.push('--eq-mid-q', String(p.eqMidQ != null ? p.eqMidQ : 1.0));
            args.push('--eq-high-cut-hz', String(p.eqHighCutHz != null ? p.eqHighCutHz : 16000.0));
          }

          const res = await runBackendOnce(repoRoot, storageDir, args, { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops op returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || sessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsUndo': {
          const sessionId = String(msg.params?.sessionId || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'undo', '--session', sessionId], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops undo returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || sessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsRedo': {
          const sessionId = String(msg.params?.sessionId || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'redo', '--session', sessionId], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops redo returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || sessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsExport': {
          const sessionId = String(msg.params?.sessionId || '');
          const outPath = String(msg.params?.outPath || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'export', '--session', sessionId, '--out', outPath], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout);
          if (!obj) throw new Error('editops export returned invalid JSON');
          lastSessionId = String(obj.session_id || obj.sessionId || sessionId || '') || lastSessionId;
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        case 'editopsClose': {
          const sessionId = String(msg.params?.sessionId || '');
          const res = await runBackendOnce(repoRoot, storageDir, ['editops', 'close', '--session', sessionId], { timeoutMs: 120000 });
          const obj = parseJsonLineBestEffort(res.stdout) || { ok: true };
          respond({ kind: 'response', id: msg.id, ok: true, result: obj });
          return;
        }
        default:
          throw new Error(`Unknown method: ${msg.method}`);
      }
    } catch (e: any) {
      respond({ kind: 'response', id: msg.id, ok: false, error: String(e?.message || e) });
    }
  });
}

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel('SÖNDBÖUND');
  context.subscriptions.push(output);

  context.subscriptions.push({
    dispose: () => {
      try {
        if (webUiProc) webUiProc.kill();
      } catch {
        // ignore
      }
      webUiProc = null;
      webUiUrl = null;
      webUiPanel = null;
    }
  });

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.openUI', async (options?: OpenUiOptions) => {
      await openEditorPanel(context, options?.wavPath);
      return { ok: true };
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.openWebUI', async (options?: OpenWebUiOptions) => {
      const repoRoot = context.extensionPath;
      const storageDir = context.globalStorageUri.fsPath;
      fs.mkdirSync(storageDir, { recursive: true });

      output.show(true);
      const url = await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'SÖNDBÖUND: Starting Web UI…',
          cancellable: false
        },
        async () => await startWebUiServer(repoRoot, storageDir, output, options)
      );

      if (webUiPanel) {
        webUiPanel.webview.html = makeWebUiHtml(webUiPanel.webview, url);
        webUiPanel.reveal(vscode.ViewColumn.One);
        return { ok: true, url };
      }

      const panel = vscode.window.createWebviewPanel(
        'sondbound.webui',
        'SÖNDBÖUND Web UI',
        vscode.ViewColumn.One,
        {
          enableScripts: false,
          retainContextWhenHidden: true,
        }
      );
      webUiPanel = panel;
      panel.webview.html = makeWebUiHtml(panel.webview, url);
      panel.onDidDispose(() => {
        if (webUiPanel === panel) webUiPanel = null;
      });
      return { ok: true, url };
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.openEditor', async () => {
      await openEditorPanel(context);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.generate', async (options?: GenerateOptions) => {
      return await vscode.commands.executeCommand('sondbound.generateSound', options);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.generateSound', async (options?: GenerateOptions) => {
      const repoRoot = context.extensionPath;
      const storageDir = context.globalStorageUri.fsPath;
      fs.mkdirSync(storageDir, { recursive: true });

      const isHeadless = !!options;
      const promptFromOptions = String(options?.prompt || '').trim();

      let prompt = promptFromOptions;
      if (!prompt) {
        const picked = await vscode.window.showInputBox({
          title: 'SÖNDBÖUND: Generate Sound',
          prompt: 'Enter a text prompt (e.g. “coin pickup”)',
          validateInput: (v) => String(v || '').trim().length ? undefined : 'Prompt is required'
        });
        if (!picked) return { ok: false, error: 'Prompt is required' };
        prompt = picked;
      }

      let engineLabel = String(options?.engine || '').trim();
      if (!engineLabel) {
        if (isHeadless) {
          engineLabel = defaultEngineFromConfig();
        } else {
          const engine = await vscode.window.showQuickPick(
            [
              { label: 'rfxgen', description: 'Fast procedural (needs rfxgen.exe)' },
              { label: 'diffusers', description: 'AI (needs torch/diffusers models)' },
              { label: 'stable_audio_open', description: 'AI (needs HF model access)' },
              { label: 'replicate', description: 'API (needs key/config)' },
              { label: 'samplelib', description: 'Remix/sample workflow' },
              { label: 'synth', description: 'DSP synth' },
              { label: 'layered', description: 'Layered' },
              { label: 'hybrid', description: 'Hybrid' },
            ],
            {
              title: 'Engine',
              canPickMany: false
            }
          );
          if (!engine) return { ok: false, error: 'Engine is required' };
          engineLabel = engine.label;
        }
      }

      let seconds = options?.seconds;
      if (seconds != null) {
        const n = Number(seconds);
        if (!Number.isFinite(n) || n <= 0) {
          return { ok: false, error: 'seconds must be a positive number' };
        }
        seconds = n;
      } else if (isHeadless) {
        seconds = defaultSecondsFromConfig();
      } else {
        const secondsStr = await vscode.window.showInputBox({
          title: 'Duration (seconds)',
          prompt: `Optional. Leave blank for default (${defaultSecondsFromConfig()}).`,
          validateInput: (v) => {
            const t = String(v || '').trim();
            if (!t) return undefined;
            const n = Number(t);
            if (!Number.isFinite(n) || n <= 0) return 'Enter a positive number';
            return undefined;
          }
        });
        seconds = secondsStr && String(secondsStr).trim() ? Number(secondsStr) : undefined;
      }

      const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri;
      const defaultOutName = `${slugifyFileStem(prompt)}.wav`;
      const defaultOutUri = wsFolder
        ? vscode.Uri.joinPath(wsFolder, defaultOutputSubdirFromConfig(), defaultOutName)
        : vscode.Uri.joinPath(context.globalStorageUri, defaultOutName);

      let outUri: vscode.Uri | undefined;
      const outputPath = String(options?.outputPath || '').trim();
      if (outputPath) {
        outUri = vscode.Uri.file(outputPath);
      } else if (isHeadless) {
        outUri = defaultOutUri;
      } else {
        const pickedOut = await vscode.window.showSaveDialog({
          title: 'Output WAV path',
          defaultUri: defaultOutUri,
          filters: { WAV: ['wav'] }
        });
        if (!pickedOut) return { ok: false, error: 'Output path is required' };
        outUri = pickedOut;
      }

      await vscode.workspace.fs.createDirectory(vscode.Uri.file(path.dirname(outUri.fsPath)));

      const post = options?.post ?? (isHeadless ? defaultPostFromConfig() : true);
      const edit = options?.edit ?? false;

      if (!isHeadless) output.show(true);
      output.appendLine(`[generate] engine=${engineLabel} seconds=${seconds ?? '(default)'} out=${outUri.fsPath}`);
      output.appendLine(`[generate] prompt: ${prompt}`);

      try {
        const run = async () => {
          const args: string[] = ['generate', '--engine', engineLabel, '--prompt', prompt, '--out', outUri.fsPath];
          if (post) args.push('--post');
          if (seconds != null) args.push('--seconds', String(seconds));

          const res = await runBackendOnce(repoRoot, storageDir, args);
          if (res.stdout) output.appendLine(res.stdout.trimEnd());
          if (res.stderr) output.appendLine(res.stderr.trimEnd());
        };

        if (isHeadless) {
          await run();
        } else {
          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'SÖNDBÖUND: Generating sound…',
              cancellable: false
            },
            run
          );
        }

        if (edit) {
          await openEditorPanel(context, outUri.fsPath);
        } else if (!isHeadless) {
          const choice = await vscode.window.showInformationMessage(
            'SÖNDBÖUND: Generated WAV.',
            'Open in Editor',
            'Reveal in Explorer'
          );
          if (choice === 'Open in Editor') {
            await openEditorPanel(context, outUri.fsPath);
          } else if (choice === 'Reveal in Explorer') {
            await vscode.commands.executeCommand('revealFileInOS', outUri);
          }
        }

        return {
          ok: true,
          outputPath: outUri.fsPath,
          engine: engineLabel,
          seconds: seconds ?? null,
          post,
        };
      } catch (e: any) {
        const detail = formatBackendError(e);
        output.appendLine(detail);
        if (!isHeadless) {
          const choice = await vscode.window.showErrorMessage(
            'SÖNDBÖUND: Generate failed. (See Output for details)',
            'Show Output'
          );
          if (choice === 'Show Output') output.show(true);
        }
        return { ok: false, error: String(e?.message || e) };
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('sondbound.exportPack', async (options?: ExportPackOptions) => {
      const repoRoot = context.extensionPath;
      const storageDir = context.globalStorageUri.fsPath;
      fs.mkdirSync(storageDir, { recursive: true });

      const isHeadless = !!options;

      let manifestPath = String(options?.manifestPath || '').trim();
      if (!manifestPath) {
        const manifestPick = await vscode.window.showOpenDialog({
          title: 'Select manifest (.json/.csv)',
          canSelectMany: false,
          filters: { Manifest: ['json', 'csv'] }
        });
        if (!manifestPick || !manifestPick[0]) return { ok: false, error: 'Manifest path is required' };
        manifestPath = manifestPick[0].fsPath;
      }

      const wsFolder = vscode.workspace.workspaceFolders?.[0]?.uri;
      const defaultZipUri = wsFolder
        ? vscode.Uri.joinPath(wsFolder, defaultOutputSubdirFromConfig(), 'resourcepack.zip')
        : vscode.Uri.joinPath(context.globalStorageUri, 'resourcepack.zip');

      let zipUri: vscode.Uri | undefined;
      const zipPath = String(options?.zipPath || '').trim();
      if (zipPath) {
        zipUri = vscode.Uri.file(zipPath);
      } else if (isHeadless) {
        zipUri = defaultZipUri;
      } else {
        const pickedZip = await vscode.window.showSaveDialog({
          title: 'Output ZIP path',
          defaultUri: defaultZipUri,
          filters: { ZIP: ['zip'] }
        });
        if (!pickedZip) return { ok: false, error: 'ZIP path is required' };
        zipUri = pickedZip;
      }

      await vscode.workspace.fs.createDirectory(vscode.Uri.file(path.dirname(zipUri.fsPath)));

      if (!isHeadless) output.show(true);
      output.appendLine(`[pack] manifest=${manifestPath}`);
      output.appendLine(`[pack] zip=${zipUri.fsPath}`);

      try {
        const run = async () => {
          const res = await runPythonModuleOnce(repoRoot, storageDir, 'soundgen.batch', [
            '--manifest', manifestPath,
            '--zip', zipUri.fsPath
          ]);
          if (res.stdout) output.appendLine(res.stdout.trimEnd());
          if (res.stderr) output.appendLine(res.stderr.trimEnd());
        };

        if (isHeadless) {
          await run();
        } else {
          await vscode.window.withProgress(
            {
              location: vscode.ProgressLocation.Notification,
              title: 'SÖNDBÖUND: Exporting pack…',
              cancellable: false
            },
            run
          );
        }

        if (!isHeadless) {
          const choice = await vscode.window.showInformationMessage(
            'SÖNDBÖUND: Pack export complete.',
            'Reveal in Explorer'
          );
          if (choice === 'Reveal in Explorer') {
            await vscode.commands.executeCommand('revealFileInOS', zipUri);
          }
        }

        return { ok: true, zipPath: zipUri.fsPath };
      } catch (e: any) {
        const detail = formatBackendError(e);
        output.appendLine(detail);
        if (!isHeadless) {
          const choice = await vscode.window.showErrorMessage(
            'SÖNDBÖUND: Pack export failed. (See Output for details)',
            'Show Output'
          );
          if (choice === 'Show Output') output.show(true);
        }
        return { ok: false, error: String(e?.message || e) };
      }
    })
  );
}

export function deactivate() {
  // no-op
}
