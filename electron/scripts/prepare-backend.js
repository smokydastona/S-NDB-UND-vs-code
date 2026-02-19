const fs = require('fs');
const path = require('path');

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function rmrf(p) {
  if (!exists(p)) return;
  fs.rmSync(p, { recursive: true, force: true });
}

function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  // Node 16+ supports fs.cpSync
  if (typeof fs.cpSync === 'function') {
    fs.cpSync(src, dst, { recursive: true });
    return;
  }
  // Fallback: manual walk
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDir(s, d);
    } else if (entry.isFile()) {
      fs.copyFileSync(s, d);
    }
  }
}

function main() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const distDir = path.join(repoRoot, 'dist');

  // Default backend output from scripts/build_exe.ps1
  const backendFolderName = process.env.SOUNDGEN_BACKEND_FOLDER || 'SÖNDBÖUND';
  const srcBackend = path.join(distDir, backendFolderName);

  const electronDir = path.join(repoRoot, 'electron');
  const dstBackendRoot = path.join(electronDir, 'backend');
  const dstBackend = path.join(dstBackendRoot, backendFolderName);

  if (!exists(srcBackend)) {
    console.error(`[prepare-backend] Missing backend folder: ${srcBackend}`);
    console.error(`[prepare-backend] Build it first: powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1 -Clean`);
    process.exit(2);
  }

  console.log(`[prepare-backend] Copying backend from ${srcBackend} -> ${dstBackend}`);
  rmrf(dstBackend);
  copyDir(srcBackend, dstBackend);

  const exe = path.join(dstBackend, `${backendFolderName}.exe`);
  if (!exists(exe)) {
    console.warn(`[prepare-backend] Warning: expected EXE not found at ${exe}`);
  }

  console.log('[prepare-backend] Done');
}

main();
