const { app, BrowserWindow, Menu, dialog } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

let backendProc = null;
let mainWindow = null;
let isQuitting = false;
let backendLogStream = null;
let backendRestartTimer = null;

let autoUpdaterApi = null;
let didSetupAutoUpdate = false;
let manualUpdateCheckInFlight = false;

function normalizeStreamingText(chunk) {
  // Progress bars often use CR without LF.
  return String(chunk || '').replace(/\r/g, '\n');
}

function getLogDir() {
  const logDir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  return logDir;
}

function logLine(line) {
  try {
    if (!backendLogStream) {
      const logPath = path.join(getLogDir(), 'electron-main.log');
      backendLogStream = fs.createWriteStream(logPath, { flags: 'a' });
    }
    backendLogStream.write(String(line || '') + (String(line || '').endsWith('\n') ? '' : '\n'));
  } catch {
    // Best-effort only.
  }
}

function isAutoUpdateEnabled() {
  // Auto-update relies on publish metadata + GitHub releases, and is intended for packaged builds.
  if (!app.isPackaged) return false;
  const raw = String(process.env.SOUNDGEN_AUTO_UPDATE || '').trim().toLowerCase();
  if (!raw) return true;
  if (['0', 'false', 'no', 'off'].includes(raw)) return false;
  return true;
}

async function setupAutoUpdate() {
  if (didSetupAutoUpdate) return;
  didSetupAutoUpdate = true;

  if (!isAutoUpdateEnabled()) return;

  try {
    // Lazy import so dev runs don't care about updater plumbing.
    // (It is still installed in packaged builds.)
    // eslint-disable-next-line global-require
    const { autoUpdater } = require('electron-updater');
    autoUpdaterApi = autoUpdater;
  } catch (e) {
    logLine('[updater] electron-updater not available: ' + (e && e.stack ? e.stack : e));
    return;
  }

  try {
    autoUpdaterApi.logger = {
      info: (m) => logLine('[updater] ' + String(m)),
      warn: (m) => logLine('[updater] ' + String(m)),
      error: (m) => logLine('[updater] ' + String(m)),
      debug: (m) => logLine('[updater] ' + String(m))
    };
  } catch {
    // best-effort
  }

  autoUpdaterApi.autoDownload = true;
  autoUpdaterApi.autoInstallOnAppQuit = true;

  autoUpdaterApi.on('error', async (err) => {
    const msg = err && err.message ? err.message : String(err);
    logLine('[updater] error: ' + msg);
    if (manualUpdateCheckInFlight) {
      manualUpdateCheckInFlight = false;
      try {
        await dialog.showMessageBox({
          type: 'error',
          buttons: ['OK'],
          title: 'Update check failed',
          message: 'Could not check for updates.',
          detail: msg
        });
      } catch {}
    }
  });

  autoUpdaterApi.on('update-available', (info) => {
    logLine('[updater] update available: ' + JSON.stringify({ version: info && info.version, files: info && info.files && info.files.length }));
  });

  autoUpdaterApi.on('update-not-available', async () => {
    logLine('[updater] no update available');
    if (manualUpdateCheckInFlight) {
      manualUpdateCheckInFlight = false;
      try {
        await dialog.showMessageBox({
          type: 'info',
          buttons: ['OK'],
          title: 'No updates',
          message: 'You are up to date.'
        });
      } catch {}
    }
  });

  autoUpdaterApi.on('download-progress', (p) => {
    // Keep logs small but helpful.
    const percent = p && typeof p.percent === 'number' ? Math.round(p.percent) : null;
    if (percent !== null) logLine('[updater] download ' + percent + '%');
  });

  autoUpdaterApi.on('update-downloaded', async () => {
    logLine('[updater] update downloaded');
    try {
      const choice = await dialog.showMessageBox({
        type: 'question',
        buttons: ['Restart now', 'Later'],
        defaultId: 0,
        cancelId: 1,
        title: 'Update ready',
        message: 'An update was downloaded.',
        detail: 'Restart SÖNDBÖUND to install it.'
      });
      if (choice.response === 0) {
        try {
          autoUpdaterApi.quitAndInstall();
        } catch (e) {
          logLine('[updater] quitAndInstall failed: ' + (e && e.stack ? e.stack : e));
        }
      }
    } catch {
      // best-effort
    }
  });

  // Kick off a background update check.
  try {
    await autoUpdaterApi.checkForUpdates();
  } catch (e) {
    logLine('[updater] initial check failed: ' + (e && e.stack ? e.stack : e));
  }
}

async function checkForUpdatesInteractive() {
  if (!autoUpdaterApi) {
    try {
      await dialog.showMessageBox({
        type: 'info',
        buttons: ['OK'],
        title: 'Updates unavailable',
        message: 'Auto-update is not enabled for this build.'
      });
    } catch {}
    return;
  }

  manualUpdateCheckInFlight = true;
  try {
    await autoUpdaterApi.checkForUpdates();
  } catch (e) {
    manualUpdateCheckInFlight = false;
    const msg = e && e.message ? e.message : String(e);
    try {
      await dialog.showMessageBox({
        type: 'error',
        buttons: ['OK'],
        title: 'Update check failed',
        message: 'Could not check for updates.',
        detail: msg
      });
    } catch {}
  }
}

function resolveBundledBackendExe() {
  // When packaged by electron-builder, extraResources land under process.resourcesPath.
  // We copy the PyInstaller onedir folder into: resources/backend/SÖNDBÖUND/
  const backendFolderName = process.env.SOUNDGEN_BACKEND_FOLDER || 'SÖNDBÖUND';
  const exeName = `${backendFolderName}.exe`;
  const exePath = path.join(process.resourcesPath, 'backend', backendFolderName, exeName);
  if (fs.existsSync(exePath)) {
    return { exePath, backendFolderName };
  }
  return null;
}

function resolveBackendCommand() {
  // Preferred: use the repo venv if present.
  const repoRoot = path.resolve(__dirname, '..');
  const venvPythonWin = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  const venvPythonPosix = path.join(repoRoot, '.venv', 'bin', 'python');

  const envPython = process.env.SOUNDGEN_PYTHON;
  if (envPython && envPython.trim()) {
    return { cmd: envPython.trim(), argsPrefix: [] };
  }

  if (process.platform === 'win32' && fs.existsSync(venvPythonWin)) {
    return { cmd: venvPythonWin, argsPrefix: [] };
  }
  if (process.platform !== 'win32' && fs.existsSync(venvPythonPosix)) {
    return { cmd: venvPythonPosix, argsPrefix: [] };
  }

  // Fallback: whatever is on PATH.
  return { cmd: 'python', argsPrefix: [] };
}

function resolveBackendRunner() {
  const repoRoot = path.resolve(__dirname, '..');
  const dataDir = app.getPath('userData');

  if (app.isPackaged) {
    const bundled = resolveBundledBackendExe();
    if (!bundled) {
      throw new Error('Bundled backend EXE not found in resources. Did you run `npm run prep-backend` before building?');
    }
    return {
      cmd: bundled.exePath,
      argsPrefix: [],
      cwd: path.dirname(bundled.exePath),
      env: { ...process.env, SOUNDGEN_DATA_DIR: dataDir, GRADIO_ANALYTICS_ENABLED: 'False' }
    };
  }

  const resolved = resolveBackendCommand();
  return {
    cmd: resolved.cmd,
    argsPrefix: ['-m', 'soundgen.app'],
    cwd: repoRoot,
    env: { ...process.env, SOUNDGEN_DATA_DIR: dataDir, GRADIO_ANALYTICS_ENABLED: 'False' }
  };
}

function runBackendOnce(args, { timeoutMs = 0 } = {}) {
  const runner = resolveBackendRunner();
  const fullArgs = [...(runner.argsPrefix || []), ...(args || [])];

  return new Promise((resolve, reject) => {
    const proc = spawn(runner.cmd, fullArgs, {
      cwd: runner.cwd,
      env: runner.env,
      windowsHide: true
    });

    let stdout = '';
    let stderr = '';
    proc.stdout.setEncoding('utf8');
    proc.stderr.setEncoding('utf8');
    proc.stdout.on('data', (c) => { stdout += normalizeStreamingText(c); });
    proc.stderr.on('data', (c) => { stderr += normalizeStreamingText(c); });

    let timer = null;
    if (timeoutMs && timeoutMs > 0) {
      timer = setTimeout(() => {
        try { proc.kill(); } catch {}
        reject(new Error('Timed out running backend command.'));
      }, timeoutMs);
    }

    proc.on('exit', (code) => {
      if (timer) clearTimeout(timer);
      if (code === 0) return resolve({ stdout, stderr });
      const err = new Error(`Backend command failed (exit ${code}).`);
      err.stdout = stdout;
      err.stderr = stderr;
      reject(err);
    });
  });
}

function createProgressWindow() {
  const win = new BrowserWindow({
    width: 760,
    height: 520,
    title: 'SÖNDBÖUND — Downloading models',
    show: true,
    webPreferences: { sandbox: true }
  });

  const html = `<!doctype html><html><head><meta charset="utf-8" />
  <title>Downloading models</title>
  <style>body{margin:12px;font-family:Consolas,monospace;white-space:pre-wrap;}#log{white-space:pre-wrap;}</style>
  </head><body><div id="log">Preparing downloads...\n</div></body></html>`;

  win.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html));

  async function append(text) {
    if (!win || win.isDestroyed()) return;
    const payload = JSON.stringify(String(text || ''));
    try {
      await win.webContents.executeJavaScript(
        `(function(){var el=document.getElementById('log'); el.textContent += ${payload}; window.scrollTo(0, document.body.scrollHeight);})()`
      );
    } catch {
      // best-effort
    }
  }

  return { win, append };
}

async function ensureModelsReady() {
  const skip = String(process.env.SOUNDGEN_SKIP_MODEL_CHECK || '').trim();
  const allowDev = String(process.env.SOUNDGEN_MODEL_CHECK_IN_DEV || '').trim() === '1';
  if (skip === '1') return;
  if (!app.isPackaged && !allowDev) return;

  let missing = [];
  try {
    const res = await runBackendOnce(['models', 'missing', '--json'], { timeoutMs: 60000 });
    const t = String(res.stdout || '').trim();
    missing = t ? JSON.parse(t) : [];
  } catch (e) {
    logLine('[models] missing-check failed (continuing): ' + (e && e.stack ? e.stack : e));
    return;
  }

  if (!Array.isArray(missing) || missing.length === 0) return;

  const msg =
    'SÖNDBÖUND did not find the default AI models in its cache.\n\n' +
    'You can download them now (recommended) or skip for now.\n\n' +
    'Missing:\n' + missing.map((m) => `  - ${m}`).join('\n');

  const choice = await dialog.showMessageBox({
    type: 'question',
    buttons: ['Download', 'Skip'],
    defaultId: 0,
    cancelId: 1,
    title: 'Download models?',
    message: 'Models not cached',
    detail: msg
  });

  if (choice.response !== 0) return;

  const { win, append } = createProgressWindow();
  try {
    for (const rid of missing) {
      await append(`\n== Downloading ${rid} ==\n`);

      const runner = resolveBackendRunner();
      const fullArgs = [...(runner.argsPrefix || []), 'models', 'download', String(rid)];
      const proc = spawn(runner.cmd, fullArgs, {
        cwd: runner.cwd,
        env: runner.env,
        windowsHide: true
      });

      proc.stdout.setEncoding('utf8');
      proc.stderr.setEncoding('utf8');
      proc.stdout.on('data', async (c) => { await append(normalizeStreamingText(c)); });
      proc.stderr.on('data', async (c) => { await append(normalizeStreamingText(c)); });

      const exitCode = await new Promise((resolve) => proc.on('exit', (code) => resolve(code)));
      if (exitCode !== 0) {
        await append(`\n[error] download failed (exit ${exitCode})\n`);
        await dialog.showMessageBox({
          type: 'error',
          buttons: ['OK'],
          title: 'Model download failed',
          message: `Failed to download ${rid}`,
          detail: `Exit code: ${exitCode}. You can retry later via the CLI: SÖNDBÖUND.exe models download ${rid}`
        });
        break;
      }
      await append(`\n[ok] downloaded ${rid}\n`);
    }
  } finally {
    try { if (win && !win.isDestroyed()) win.close(); } catch {}
  }
}

function startBackend() {
  const repoRoot = path.resolve(__dirname, '..');

  let cmd;
  let args;
  let cwd;

  if (app.isPackaged) {
    const bundled = resolveBundledBackendExe();
    if (!bundled) {
      throw new Error('Bundled backend EXE not found in resources. Did you run `npm run prep-backend` before building?');
    }
    cmd = bundled.exePath;
    // PyInstaller EXE entrypoint: sndbund_entry.py -> soundgen.app:main
    args = ['serve', '--host', '127.0.0.1', '--port', '0', '--print-json'];
    cwd = path.dirname(cmd);
  } else {
    const resolved = resolveBackendCommand();
    cmd = resolved.cmd;
    // `python -m soundgen.app serve` runs the Gradio server for the Electron shell.
    args = ['-m', 'soundgen.app', 'serve', '--host', '127.0.0.1', '--port', '0', '--print-json'];
    cwd = repoRoot;
  }

  const logPath = path.join(getLogDir(), 'electron-backend.log');
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });

  // Ensure both Electron and Python agree on a single per-user writable directory.
  const dataDir = app.getPath('userData');

  backendProc = spawn(cmd, args, {
    cwd,
    env: {
      ...process.env,
      SOUNDGEN_DATA_DIR: dataDir,
      // Ensures the backend doesn't try to open a browser window.
      GRADIO_ANALYTICS_ENABLED: 'False'
    },
    windowsHide: true
  });

  backendProc.stdout.setEncoding('utf8');
  backendProc.stderr.setEncoding('utf8');

  backendProc.stdout.on('data', (chunk) => {
    logStream.write(chunk);
  });
  backendProc.stderr.on('data', (chunk) => {
    logStream.write(chunk);
  });

  backendProc.on('exit', (code) => {
    logStream.write(`\n[backend exited] code=${code}\n`);
    if (!isQuitting) {
      // If the backend dies, we try to restart it and reload.
      scheduleBackendRestart();
    }
  });

  return new Promise((resolve, reject) => {
    let buffer = '';
    const timeoutMs = 30000;
    const start = Date.now();

    function tryParseUrl(text) {
      // Preferred (machine-readable): a single JSON line like {"url": "http://127.0.0.1:7860", ...}
      const lines = text.split(/\r?\n/);
      for (const line of lines) {
        const t = (line || '').trim();
        if (!t) continue;
        if (t.startsWith('{') && t.endsWith('}')) {
          try {
            const obj = JSON.parse(t);
            if (obj && obj.url) return String(obj.url);
          } catch {}
        }
      }

      // Back-compat: SOUNDGEN_URL=...
      const m = text.match(/SOUNDGEN_URL=(https?:\/\/[^\s]+)/);
      if (m && m[1]) return m[1];
      return null;
    }

    const onData = (chunk) => {
      buffer += chunk;
      const url = tryParseUrl(buffer);
      if (url) {
        backendProc.stdout.off('data', onData);
        resolve(url);
      }
    };

    backendProc.stdout.on('data', onData);

    const interval = setInterval(() => {
      if (!backendProc || backendProc.killed) {
        clearInterval(interval);
        reject(new Error('Backend process terminated before URL was printed.'));
        return;
      }
      if (Date.now() - start > timeoutMs) {
        clearInterval(interval);
        reject(new Error(`Timed out waiting for backend URL. See log: ${logPath}`));
      }
    }, 250);
  });
}

function scheduleBackendRestart() {
  if (backendRestartTimer) return;
  backendRestartTimer = setTimeout(async () => {
    backendRestartTimer = null;
    try {
      logLine('[backend] restarting...');
      const url = await startBackend();
      if (mainWindow && !mainWindow.isDestroyed()) {
        await mainWindow.loadURL(url);
      }
    } catch (e) {
      logLine(`[backend] restart failed: ${e && e.stack ? e.stack : e}`);
      // Give the user something actionable.
      if (mainWindow && !mainWindow.isDestroyed()) {
        const msg = String(e && e.message ? e.message : e);
        await mainWindow.loadURL('data:text/plain;charset=utf-8,' + encodeURIComponent('Failed to restart backend.\n\n' + msg));
      }
    }
  }, 1500);
}

function createMenu() {
  const updateItem = isAutoUpdateEnabled()
    ? [{
        label: 'Check for Updates…',
        click: () => {
          checkForUpdatesInteractive().catch((e) => logLine('[updater] interactive check failed: ' + (e && e.stack ? e.stack : e)));
        }
      }]
    : [];

  const template = [
    {
      label: 'App',
      submenu: [
        { role: 'reload' },
        { role: 'toggledevtools' },
        ...updateItem,
        { type: 'separator' },
        { role: 'quit' }
      ]
    }
  ];
  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

async function createWindow() {
  createMenu();

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    title: 'SÖNDBÖUND',
    webPreferences: {
      sandbox: true
    }
  });

  await ensureModelsReady();
  const url = await startBackend();
  await mainWindow.loadURL(url);
  mainWindow.show();
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  isQuitting = true;
  if (backendProc && !backendProc.killed) {
    try {
      backendProc.kill();
    } catch {}
  }
});

// Single-instance lock (Windows app behavior)
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

process.on('uncaughtException', (err) => {
  logLine(`[uncaughtException] ${err && err.stack ? err.stack : err}`);
  try {
    dialog.showErrorBox('SÖNDBÖUND crashed', String(err && err.message ? err.message : err));
  } catch {}
});

app.whenReady().then(() => {
  setupAutoUpdate()
    .catch((e) => logLine('[updater] setup failed: ' + (e && e.stack ? e.stack : e)))
    .finally(() => {
      createWindow().catch((e) => {
    // If backend failed, show a basic error page.
    const msg = String(e && e.message ? e.message : e);
    mainWindow = new BrowserWindow({ width: 900, height: 600, title: 'SÖNDBÖUND (error)' });
    mainWindow.loadURL('data:text/plain;charset=utf-8,' + encodeURIComponent('Failed to start backend.\n\n' + msg));
      });
    });
});
