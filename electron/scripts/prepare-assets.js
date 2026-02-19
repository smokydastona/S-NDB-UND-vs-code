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

async function main() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const srcPng = path.join(repoRoot, '.examples', 'icon.png');

  const electronDir = path.join(repoRoot, 'electron');
  const buildDir = path.join(electronDir, 'build');
  const outIco = path.join(buildDir, 'icon.ico');

  fs.mkdirSync(buildDir, { recursive: true });

  if (!exists(srcPng)) {
    console.warn(`[prepare-assets] Missing icon source PNG: ${srcPng}`);
    return;
  }

  // Lazy import so `npm install` doesn't error on require during non-build flows.
  const pngToIco = require('png-to-ico');

  console.log(`[prepare-assets] Generating ICO: ${srcPng} -> ${outIco}`);
  const buf = await pngToIco(srcPng);
  fs.writeFileSync(outIco, buf);

  console.log('[prepare-assets] Done');
}

main().catch((e) => {
  console.error('[prepare-assets] Failed:', e);
  process.exit(1);
});
