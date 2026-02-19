const { app, BrowserWindow, Menu } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

let backendProc = null;
let mainWindow = null;

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
    args = ['serve', '--host', '127.0.0.1', '--port', '0', '--print-json'];
    cwd = path.dirname(cmd);
  } else {
    const resolved = resolveBackendCommand();
    cmd = resolved.cmd;
    args = ['-m', 'soundgen.app', 'serve', '--host', '127.0.0.1', '--port', '0', '--print-json'];
    cwd = repoRoot;
  }

  const logDir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  const logPath = path.join(logDir, 'electron-backend.log');
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });

  backendProc = spawn(cmd, args, {
    cwd,
    env: {
      ...process.env,
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

function createMenu() {
  const template = [
    {
      label: 'App',
      submenu: [
        { role: 'reload' },
        { role: 'toggledevtools' },
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

  const url = await startBackend();
  await mainWindow.loadURL(url);
  mainWindow.show();
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (backendProc && !backendProc.killed) {
    try {
      backendProc.kill();
    } catch {}
  }
});

app.whenReady().then(() => {
  createWindow().catch((e) => {
    // If backend failed, show a basic error page.
    const msg = String(e && e.message ? e.message : e);
    mainWindow = new BrowserWindow({ width: 900, height: 600, title: 'SÖNDBÖUND (error)' });
    mainWindow.loadURL(
      'data:text/plain;charset=utf-8,' + encodeURIComponent('Failed to start backend.\n\n' + msg)
    );
  });
});
