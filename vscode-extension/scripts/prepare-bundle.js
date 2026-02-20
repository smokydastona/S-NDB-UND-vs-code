const fs = require('fs');
const path = require('path');

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function copyDir(src, dst) {
  if (!fs.existsSync(src)) {
    throw new Error(`Missing source directory: ${src}`);
  }
  ensureDir(path.dirname(dst));
  fs.cpSync(src, dst, { recursive: true, force: true });
}

function copyFile(src, dst) {
  if (!fs.existsSync(src)) {
    throw new Error(`Missing source file: ${src}`);
  }
  ensureDir(path.dirname(dst));
  fs.copyFileSync(src, dst);
}

function main() {
  const extRoot = path.resolve(__dirname, '..');
  const repoRoot = path.resolve(extRoot, '..');

  // Backend bundle (Python)
  copyDir(path.join(repoRoot, 'src', 'soundgen'), path.join(extRoot, 'backend', 'src', 'soundgen'));
  copyFile(path.join(repoRoot, 'requirements.txt'), path.join(extRoot, 'backend', 'requirements.txt'));
  copyFile(path.join(repoRoot, 'pyproject.toml'), path.join(extRoot, 'backend', 'pyproject.toml'));

  // Configs + example manifest
  copyDir(path.join(repoRoot, 'configs'), path.join(extRoot, 'configs'));
  copyFile(path.join(repoRoot, 'example_manifest.json'), path.join(extRoot, 'example_manifest.json'));

  // Webview/editor assets
  copyDir(path.join(repoRoot, 'webview', 'editor'), path.join(extRoot, 'electron', 'editor'));

  // Marketplace icon
  copyFile(path.join(repoRoot, '.examples', 'icon.png'), path.join(extRoot, '.examples', 'icon.png'));

  console.log('[prepare-bundle] OK');
}

main();
