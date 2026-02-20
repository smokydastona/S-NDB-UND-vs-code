/* global soundgenEditor */

const $ = (id) => document.getElementById(id);

const ui = {
  openBtn: $("openBtn"),
  loadClapBtn: $("loadClapBtn"),
  loadLv2Btn: $("loadLv2Btn"),
  pluginPreviewChk: $("pluginPreviewChk"),
  trimBtn: $("trimBtn"),
  cutBtn: $("cutBtn"),
  copyBtn: $("copyBtn"),
  pasteBtn: $("pasteBtn"),
  deleteBtn: $("deleteBtn"),
  silenceBtn: $("silenceBtn"),
  fadeInBtn: $("fadeInBtn"),
  fadeOutBtn: $("fadeOutBtn"),
  normBtn: $("normBtn"),
  revBtn: $("revBtn"),
  pitchBtn: $("pitchBtn"),
  eqBtn: $("eqBtn"),
  undoBtn: $("undoBtn"),
  redoBtn: $("redoBtn"),
  exportBtn: $("exportBtn"),
  playBtn: $("playBtn"),
  stopBtn: $("stopBtn"),
  loopChk: $("loopChk"),
  fileInfo: $("fileInfo"),
  history: $("history"),
  status: $("status"),
  timeLabel: $("timeLabel"),
  canvas: $("wave"),
};

const state = {
  sessionId: null,
  sampleRate: 44100,
  currentWavPath: null,
  audioBuffer: null,
  samples: null,
  plugins: {
    clapPath: null,
    clapPluginId: null,
    lv2BundlePath: null,
  },
  preview: {
    forWavPath: null,
    wavPath: null,
    audioBuffer: null,
    samples: null,
    rendering: false,
  },
  viewStart: 0,
  viewEnd: 0,
  cursor: 0,
  selecting: false,
  selectionStart: null,
  selectionEnd: null,
  scrubbing: false,
  lastScrubAt: 0,
  scrubSource: null,
  panMode: false,
  panAnchorX: 0,
  panAnchorViewStart: 0,
  panAnchorViewEnd: 0,
  audioCtx: null,
  source: null,
  playing: false,
};

function setStatus(text) {
  ui.status.textContent = String(text || "");
}

function setHintIfIdle() {
  if (String(ui.status.textContent || '').trim()) return;
  setStatus('Tip: Alt+drag scrub | Shift+drag pan | Wheel zoom | Drag select | Space play');
}

function shortPath(p) {
  return String(p || '').split(/[\\/]/).filter(Boolean).slice(-1)[0] || String(p || '');
}

function isPluginPreviewEnabled() {
  return !!(ui.pluginPreviewChk && ui.pluginPreviewChk.checked);
}

function invalidatePluginPreview() {
  state.preview.forWavPath = null;
  state.preview.wavPath = null;
  state.preview.audioBuffer = null;
  state.preview.samples = null;
  state.preview.rendering = false;
}

function derivePreviewOutPath(inWavPath) {
  const p = String(inWavPath || '');
  if (!p) return '';
  if (p.toLowerCase().endsWith('.wav')) return p.replace(/\.wav$/i, '.preview.wav');
  return p + '.preview.wav';
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function selectionRange() {
  if (state.selectionStart == null || state.selectionEnd == null) return null;
  const a = Math.min(state.selectionStart, state.selectionEnd);
  const b = Math.max(state.selectionStart, state.selectionEnd);
  if (b <= a) return null;
  return [a, b];
}

function enableEditing(enabled) {
  for (const b of [
    ui.loadClapBtn,
    ui.loadLv2Btn,
    ui.trimBtn,
    ui.cutBtn,
    ui.copyBtn,
    ui.pasteBtn,
    ui.deleteBtn,
    ui.silenceBtn,
    ui.fadeInBtn,
    ui.fadeOutBtn,
    ui.normBtn,
    ui.revBtn,
    ui.pitchBtn,
    ui.eqBtn,
    ui.undoBtn,
    ui.redoBtn,
    ui.exportBtn,
    ui.playBtn,
    ui.stopBtn
  ]) {
    if (!b) continue;
    b.disabled = !enabled;
  }

  if (ui.pluginPreviewChk) ui.pluginPreviewChk.disabled = !enabled;
}

function fmtTime(samples) {
  const sr = state.sampleRate || 1;
  const t = samples / sr;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

function updateTimeLabel() {
  const sel = selectionRange();
  if (sel) {
    ui.timeLabel.textContent = `Cursor ${fmtTime(state.cursor)} | Sel ${fmtTime(sel[0])} - ${fmtTime(sel[1])}`;
  } else {
    ui.timeLabel.textContent = `Cursor ${fmtTime(state.cursor)}`;
  }
}

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

async function decodeWavToMono(wavPath) {
  const b64 = await soundgenEditor.readFileBase64(wavPath);
  const buf = base64ToArrayBuffer(b64);
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await state.audioCtx.decodeAudioData(buf.slice(0));

  // Ensure mono.
  let mono;
  if (decoded.numberOfChannels === 1) {
    mono = decoded.getChannelData(0);
  } else {
    const len = decoded.length;
    mono = new Float32Array(len);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const d = decoded.getChannelData(ch);
      for (let i = 0; i < len; i++) mono[i] += d[i] / decoded.numberOfChannels;
    }
  }

  return { decoded, mono, sampleRate: decoded.sampleRate };
}

async function loadAudioFromWavPath(wavPath) {
  const { decoded, mono, sampleRate } = await decodeWavToMono(wavPath);

  state.audioBuffer = decoded;
  state.samples = mono;
  state.sampleRate = sampleRate;
  state.viewStart = 0;
  state.viewEnd = mono.length;
  state.cursor = 0;
  state.selectionStart = null;
  state.selectionEnd = null;
  resizeCanvas();
  draw();
  updateTimeLabel();
}

async function loadPreviewFromWavPath(wavPath, forWavPath) {
  const { decoded, mono } = await decodeWavToMono(wavPath);
  state.preview.forWavPath = String(forWavPath || '');
  state.preview.wavPath = String(wavPath || '');
  state.preview.audioBuffer = decoded;
  state.preview.samples = mono;
}

async function ensurePluginPreviewReady() {
  if (!isPluginPreviewEnabled()) return false;
  if (!state.currentWavPath) return false;
  if (state.preview.rendering) return false;

  if (state.plugins.lv2BundlePath) {
    setStatus('LV2 selected, but LV2 preview is not implemented yet.');
    return false;
  }

  if (!state.plugins.clapPath) {
    setStatus('Preview enabled, but no CLAP plugin loaded.');
    return false;
  }

  if (!state.plugins.clapPluginId) {
    setStatus('Preview enabled, but no CLAP plugin is selected.');
    return false;
  }

  if (state.preview.audioBuffer && state.preview.forWavPath === state.currentWavPath) {
    return true;
  }

  const outWav = derivePreviewOutPath(state.currentWavPath);
  if (!outWav) return false;

  state.preview.rendering = true;
  try {
    setStatus(`Rendering CLAP preview (${shortPath(state.plugins.clapPath)})…`);
    await soundgenEditor.clapRenderPreview({
      inWav: state.currentWavPath,
      outWav,
      pluginPath: state.plugins.clapPath,
      pluginId: state.plugins.clapPluginId,
    });
    await loadPreviewFromWavPath(outWav, state.currentWavPath);
    setStatus('');
    setHintIfIdle();
    return true;
  } catch (e) {
    invalidatePluginPreview();
    setStatus('Plugin preview failed: ' + String(e && e.message ? e.message : e));
    return false;
  } finally {
    state.preview.rendering = false;
  }
}

function activePlaybackBuffer() {
  if (isPluginPreviewEnabled() && state.preview.audioBuffer) return state.preview.audioBuffer;
  return state.audioBuffer;
}

function resizeCanvas() {
  const rect = ui.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  ui.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  ui.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
}

function xToSample(clientX) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = clamp(clientX - rect.left, 0, rect.width);
  const frac = rect.width ? (x / rect.width) : 0;
  const s = state.viewStart + frac * (state.viewEnd - state.viewStart);
  return Math.round(clamp(s, 0, (state.samples ? state.samples.length : 1) - 1));
}

function draw() {
  const ctx = ui.canvas.getContext('2d');
  if (!ctx) return;

  const w = ui.canvas.width;
  const h = ui.canvas.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, w, h);

  if (!state.samples) {
    ctx.fillStyle = '#333';
    ctx.font = `${Math.floor(14 * (window.devicePixelRatio || 1))}px system-ui`;
    ctx.fillText('Open a WAV to start.', 12, 24);
    return;
  }

  const n = state.samples.length;
  const a = clamp(state.viewStart, 0, n);
  const b = clamp(state.viewEnd, 0, n);
  const span = Math.max(1, b - a);

  // Selection shading.
  const sel = selectionRange();
  if (sel) {
    const s0 = clamp(sel[0], a, b);
    const s1 = clamp(sel[1], a, b);
    const x0 = Math.floor(((s0 - a) / span) * w);
    const x1 = Math.floor(((s1 - a) / span) * w);
    ctx.fillStyle = 'rgba(70,130,180,0.18)';
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
  }

  // Waveform.
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 1;
  ctx.beginPath();

  const mid = Math.floor(h / 2);
  for (let x = 0; x < w; x++) {
    const s0 = Math.floor(a + (x / w) * span);
    const s1 = Math.floor(a + ((x + 1) / w) * span);
    const lo = Math.max(a, Math.min(s0, s1));
    const hi = Math.max(lo + 1, Math.min(b, Math.max(s0, s1)));

    let mn = 1.0;
    let mx = -1.0;
    for (let i = lo; i < hi; i++) {
      const v = state.samples[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }

    const y0 = mid - mn * (mid - 1);
    const y1 = mid - mx * (mid - 1);
    ctx.moveTo(x + 0.5, y0);
    ctx.lineTo(x + 0.5, y1);
  }
  ctx.stroke();

  // Cursor.
  const cx = Math.floor(((clamp(state.cursor, a, b) - a) / span) * w);
  ctx.strokeStyle = 'rgba(220,20,60,0.9)';
  ctx.beginPath();
  ctx.moveTo(cx + 0.5, 0);
  ctx.lineTo(cx + 0.5, h);
  ctx.stroke();
}

function stopPlayback() {
  try {
    if (state.source) state.source.stop();
  } catch {}
  state.source = null;
  state.playing = false;
}

function stopScrub() {
  try {
    if (state.scrubSource) state.scrubSource.stop();
  } catch {}
  state.scrubSource = null;
}

async function scrubPreviewAt(sampleIndex) {
  if (!state.audioCtx || !state.audioBuffer) return;
  const now = performance.now();
  if (now - state.lastScrubAt < 35) return; // throttle
  state.lastScrubAt = now;

  // Ensure the AudioContext is running (some browsers require a user gesture).
  try {
    if (state.audioCtx.state !== 'running') await state.audioCtx.resume();
  } catch {}

  stopScrub();

  const sr = state.sampleRate || 1;
  const buf = activePlaybackBuffer() || state.audioBuffer;
  const t = clamp(sampleIndex / sr, 0, buf.duration);
  const dur = 0.045;

  const src = state.audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(state.audioCtx.destination);
  src.onended = () => { if (state.scrubSource === src) state.scrubSource = null; };
  state.scrubSource = src;

  try {
    // duration form can throw if duration <= 0 or start is at EOF.
    src.start(0, t, Math.max(0.01, Math.min(dur, Math.max(0, buf.duration - t))));
  } catch {
    try { src.start(0, t); } catch {}
  }
}

async function playFromCursor() {
  if (!state.audioCtx || !state.audioBuffer) return;
  stopPlayback();

  // If preview is enabled, attempt to render it once per snapshot.
  try {
    await ensurePluginPreviewReady();
  } catch {
    // best-effort
  }

  const buf = activePlaybackBuffer() || state.audioBuffer;

  const src = state.audioCtx.createBufferSource();
  src.buffer = buf;

  const sr = state.sampleRate || 1;
  const startTime = clamp(state.cursor / sr, 0, buf.duration);

  const sel = selectionRange();
  const loop = ui.loopChk.checked && !!sel;
  if (loop) {
    const ls = clamp(sel[0] / sr, 0, buf.duration);
    const le = clamp(sel[1] / sr, 0, buf.duration);
    if (le > ls) {
      src.loop = true;
      src.loopStart = ls;
      src.loopEnd = le;
    }
  }

  src.connect(state.audioCtx.destination);
  src.onended = () => {
    if (state.source === src) {
      state.source = null;
      state.playing = false;
    }
  };

  state.source = src;
  state.playing = true;

  // If looping and cursor is outside selection, start at selection start.
  if (loop && sel) {
    const ls = sel[0] / sr;
    const le = sel[1] / sr;
    if (startTime < ls || startTime > le) {
      src.start(0, ls);
      return;
    }
  }
  src.start(0, startTime);
}

function updateButtonsFromResp(resp) {
  ui.undoBtn.disabled = !resp.can_undo;
  ui.redoBtn.disabled = !resp.can_redo;
  ui.pasteBtn.disabled = !resp.has_clipboard;
}

function renderHistory(items) {
  ui.history.innerHTML = '';
  const list = Array.isArray(items) ? items.slice(-50).reverse() : [];
  for (const it of list) {
    const div = document.createElement('div');
    div.className = 'item';
    const op = document.createElement('div');
    op.className = 'op';
    op.textContent = String(it.op || 'op');
    const params = document.createElement('div');
    params.textContent = JSON.stringify(it.params || {});
    div.appendChild(op);
    div.appendChild(params);
    ui.history.appendChild(div);
  }
}

async function refreshFromResp(resp) {
  if (!resp) return;

  const prevWav = state.currentWavPath;
  state.sampleRate = resp.sample_rate;
  state.currentWavPath = resp.current_wav;
  if (prevWav && prevWav !== state.currentWavPath) {
    invalidatePluginPreview();
  }

  enableEditing(true);

  ui.fileInfo.textContent = `SR ${resp.sample_rate} Hz | ${resp.duration_s.toFixed(3)} s | ${resp.length_samples} samples`;

  updateButtonsFromResp(resp);
  renderHistory(resp.history);

  await loadAudioFromWavPath(resp.current_wav);
  setStatus('');
  setHintIfIdle();
}

async function doOpen() {
  try {
    const path = await soundgenEditor.openWavDialog();
    if (!path) return;

    // Close previous session.
    if (state.sessionId) {
      try { await soundgenEditor.editopsClose(state.sessionId); } catch {}
      state.sessionId = null;
    }

    const resp = await soundgenEditor.editopsInit(path);
    state.sessionId = resp.session_id;
    await refreshFromResp(resp);
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e));
  }
}

async function applyOp(type, args = {}) {
  if (!state.sessionId) return;
  try {
    const sel = selectionRange();
    const payload = {
      sessionId: state.sessionId,
      type,
      start: sel ? sel[0] : null,
      end: sel ? sel[1] : null,
      cursor: state.cursor,
      ...args,
    };
    const resp = await soundgenEditor.editopsOp(payload);
    await refreshFromResp(resp);
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e));
  }
}

ui.openBtn.onclick = () => doOpen();

if (ui.loadClapBtn) {
  ui.loadClapBtn.onclick = async () => {
    try {
      const p = await soundgenEditor.pickClapPluginDialog();
      if (!p) return;
      state.plugins.clapPath = p;
      state.plugins.clapPluginId = null;
      state.plugins.lv2BundlePath = null;
      invalidatePluginPreview();

      const list = await soundgenEditor.clapListPlugins({ pluginPath: p });
      const plugins = Array.isArray(list.plugins) ? list.plugins : [];
      if (!plugins.length) {
        setStatus('CLAP loaded, but no plugin descriptors were found.');
        return;
      }

      if (plugins.length === 1) {
        state.plugins.clapPluginId = String(plugins[0].id || '');
        setStatus(`Loaded CLAP: ${shortPath(p)} | Plugin: ${plugins[0].name || plugins[0].id || '0'}`);
        return;
      }

      const lines = plugins.map((pl, i) => `${i}: ${pl.name || pl.id || '(unnamed)'}${pl.id ? ` [${pl.id}]` : ''}`);
      const idxRaw = prompt(
        `This .clap contains multiple plugins. Pick one by number:\n\n${lines.join('\n')}\n\nEnter index:`,
        '0'
      );
      if (idxRaw == null) {
        setStatus('CLAP loaded. Plugin selection canceled.');
        return;
      }
      const idx = parseInt(String(idxRaw).trim(), 10);
      if (!isFinite(idx) || idx < 0 || idx >= plugins.length) {
        setStatus('Invalid plugin index.');
        return;
      }
      state.plugins.clapPluginId = String(plugins[idx].id || '');
      setStatus(`Loaded CLAP: ${shortPath(p)} | Plugin: ${plugins[idx].name || plugins[idx].id || String(idx)}`);
    } catch (e) {
      setStatus(String(e && e.message ? e.message : e));
    }
  };
}

if (ui.loadLv2Btn) {
  ui.loadLv2Btn.onclick = async () => {
    try {
      const p = await soundgenEditor.pickLv2BundleDialog();
      if (!p) return;
      state.plugins.lv2BundlePath = p;
      state.plugins.clapPath = null;
      state.plugins.clapPluginId = null;
      invalidatePluginPreview();
      setStatus(`Loaded LV2: ${shortPath(p)} (preview not implemented yet)`);
    } catch (e) {
      setStatus(String(e && e.message ? e.message : e));
    }
  };
}

if (ui.pluginPreviewChk) {
  ui.pluginPreviewChk.onchange = () => {
    if (!isPluginPreviewEnabled()) {
      setStatus('');
      setHintIfIdle();
      return;
    }
    if (state.plugins.clapPath) {
      setStatus('Plugin preview enabled. Press Play to render preview.');
    } else if (state.plugins.lv2BundlePath) {
      setStatus('LV2 selected, but LV2 preview is not implemented yet.');
    } else {
      setStatus('Plugin preview enabled. Load a CLAP plugin to preview.');
    }
  };
}
ui.trimBtn.onclick = () => {
  const sel = selectionRange();
  if (!sel) return setStatus('Trim requires a selection.');
  applyOp('trim');
};
ui.copyBtn.onclick = () => {
  const sel = selectionRange();
  if (!sel) return setStatus('Copy requires a selection.');
  applyOp('copy');
};
ui.cutBtn.onclick = () => {
  const sel = selectionRange();
  if (!sel) return setStatus('Cut requires a selection.');
  applyOp('cut');
};
ui.deleteBtn.onclick = () => {
  const sel = selectionRange();
  if (!sel) return setStatus('Delete requires a selection.');
  applyOp('delete');
};
ui.pasteBtn.onclick = () => {
  applyOp('paste');
};
ui.silenceBtn.onclick = () => {
  const ms = parseFloat(prompt('Insert silence (ms):', '250') || '');
  if (!isFinite(ms) || ms <= 0) return;
  applyOp('silence_insert', { silenceMs: ms });
};
ui.fadeInBtn.onclick = () => {
  const ms = parseFloat(prompt('Fade in (ms):', '30') || '');
  if (!isFinite(ms)) return;
  applyOp('fade', { fadeMode: 'in', fadeMs: ms });
};
ui.fadeOutBtn.onclick = () => {
  const ms = parseFloat(prompt('Fade out (ms):', '30') || '');
  if (!isFinite(ms)) return;
  applyOp('fade', { fadeMode: 'out', fadeMs: ms });
};
ui.normBtn.onclick = () => {
  const db = parseFloat(prompt('Normalize peak (dBFS):', '-1') || '');
  if (!isFinite(db)) return;
  applyOp('normalize', { normalizePeakDb: db });
};
ui.revBtn.onclick = () => applyOp('reverse');
ui.pitchBtn.onclick = () => {
  const s = parseFloat(prompt('Pitch shift semitones (preserve duration):', '0') || '');
  if (!isFinite(s) || s === 0) return;
  applyOp('pitch', { pitchSemitones: s });
};
ui.eqBtn.onclick = () => {
  const lc = parseFloat(prompt('EQ low cut (Hz):', '80') || '');
  const mf = parseFloat(prompt('EQ mid freq (Hz):', '1200') || '');
  const mg = parseFloat(prompt('EQ mid gain (dB):', '0') || '');
  const mq = parseFloat(prompt('EQ mid Q:', '1.0') || '');
  const hc = parseFloat(prompt('EQ high cut (Hz):', '16000') || '');
  if (![lc, mf, mg, mq, hc].every((v) => isFinite(v))) return;
  applyOp('eq3', { eqLowCutHz: lc, eqMidFreqHz: mf, eqMidGainDb: mg, eqMidQ: mq, eqHighCutHz: hc });
};

ui.undoBtn.onclick = async () => {
  if (!state.sessionId) return;
  try {
    const resp = await soundgenEditor.editopsUndo(state.sessionId);
    await refreshFromResp(resp);
  } catch (e) { setStatus(String(e && e.message ? e.message : e)); }
};
ui.redoBtn.onclick = async () => {
  if (!state.sessionId) return;
  try {
    const resp = await soundgenEditor.editopsRedo(state.sessionId);
    await refreshFromResp(resp);
  } catch (e) { setStatus(String(e && e.message ? e.message : e)); }
};

ui.exportBtn.onclick = async () => {
  if (!state.sessionId) return;
  try {
    const out = await soundgenEditor.saveWavDialog();
    if (!out) return;
    const resp = await soundgenEditor.editopsExport(state.sessionId, out);
    renderHistory(resp.history);
    setStatus(`Exported: ${out}`);
  } catch (e) {
    setStatus(String(e && e.message ? e.message : e));
  }
};

ui.playBtn.onclick = () => { playFromCursor().catch((e) => setStatus(String(e && e.message ? e.message : e))); };
ui.stopBtn.onclick = () => stopPlayback();

window.addEventListener('resize', () => {
  resizeCanvas();
  draw();
});

ui.canvas.addEventListener('mousedown', (ev) => {
  if (!state.samples) return;
  if (ev.shiftKey) {
    state.panMode = true;
    state.panAnchorX = ev.clientX;
    state.panAnchorViewStart = state.viewStart;
    state.panAnchorViewEnd = state.viewEnd;
    return;
  }
  if (ev.altKey) {
    state.scrubbing = true;
    const s = xToSample(ev.clientX);
    state.cursor = s;
    updateTimeLabel();
    draw();
    scrubPreviewAt(s);
    return;
  }
  state.selecting = true;
  const s = xToSample(ev.clientX);
  state.selectionStart = s;
  state.selectionEnd = s;
  state.cursor = s;
  updateTimeLabel();
  draw();
});

ui.canvas.addEventListener('mousemove', (ev) => {
  if (!state.samples) return;
  if (state.panMode) {
    const rect = ui.canvas.getBoundingClientRect();
    const dx = (ev.clientX - state.panAnchorX) / rect.width;
    const span = state.panAnchorViewEnd - state.panAnchorViewStart;
    const shift = Math.round(-dx * span);

    const n = state.samples.length;
    let a = clamp(state.panAnchorViewStart + shift, 0, n);
    let b = clamp(state.panAnchorViewEnd + shift, 0, n);
    if (b - a < 16) b = a + 16;
    if (b > n) { b = n; a = Math.max(0, b - span); }

    state.viewStart = a;
    state.viewEnd = b;
    draw();
    return;
  }

  if (state.scrubbing) {
    const s = xToSample(ev.clientX);
    state.cursor = s;
    updateTimeLabel();
    draw();
    scrubPreviewAt(s);
    return;
  }

  if (!state.selecting) return;
  const s = xToSample(ev.clientX);
  state.selectionEnd = s;
  state.cursor = s;
  updateTimeLabel();
  draw();
});

ui.canvas.addEventListener('mouseup', (ev) => {
  if (state.panMode) {
    state.panMode = false;
    return;
  }
  if (state.scrubbing) {
    state.scrubbing = false;
    stopScrub();
    return;
  }
  if (!state.selecting) return;
  state.selecting = false;
  const s = xToSample(ev.clientX);
  state.selectionEnd = s;
  state.cursor = s;
  updateTimeLabel();
  draw();
});

ui.canvas.addEventListener('mouseleave', () => {
  state.selecting = false;
  state.panMode = false;
  if (state.scrubbing) {
    state.scrubbing = false;
    stopScrub();
  }
});

ui.canvas.addEventListener('wheel', (ev) => {
  if (!state.samples) return;
  ev.preventDefault();

  const n = state.samples.length;
  const zoomIn = ev.deltaY < 0;
  const factor = zoomIn ? 0.8 : 1.25;
  const center = xToSample(ev.clientX);

  const curSpan = Math.max(16, state.viewEnd - state.viewStart);
  let newSpan = Math.round(curSpan * factor);
  newSpan = clamp(newSpan, 16, n);

  let a = Math.round(center - newSpan / 2);
  let b = a + newSpan;
  if (a < 0) { a = 0; b = newSpan; }
  if (b > n) { b = n; a = Math.max(0, b - newSpan); }

  state.viewStart = a;
  state.viewEnd = b;
  draw();
}, { passive: false });

window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space') {
    ev.preventDefault();
    playFromCursor().catch(() => {});
    return;
  }
  if (ev.key === 'Delete' || ev.key === 'Backspace') {
    const sel = selectionRange();
    if (!sel) return;
    ev.preventDefault();
    ui.deleteBtn.click();
    return;
  }
  if (ev.ctrlKey && ev.key.toLowerCase() === 'c') {
    const sel = selectionRange();
    if (!sel) return;
    ev.preventDefault();
    ui.copyBtn.click();
    return;
  }
  if (ev.ctrlKey && ev.key.toLowerCase() === 'x') {
    const sel = selectionRange();
    if (!sel) return;
    ev.preventDefault();
    ui.cutBtn.click();
    return;
  }
  if (ev.ctrlKey && ev.key.toLowerCase() === 'v') {
    ev.preventDefault();
    ui.pasteBtn.click();
    return;
  }
  if (ev.ctrlKey && ev.key.toLowerCase() === 'z') {
    ev.preventDefault();
    ui.undoBtn.click();
    return;
  }
  if (ev.ctrlKey && ev.key.toLowerCase() === 'y') {
    ev.preventDefault();
    ui.redoBtn.click();
  }
});

window.addEventListener('beforeunload', async () => {
  stopPlayback();
  stopScrub();
  if (state.sessionId) {
    try { await soundgenEditor.editopsClose(state.sessionId); } catch {}
  }
});

// Initial state.
enableEditing(false);
resizeCanvas();
draw();
updateTimeLabel();
setHintIfIdle();
