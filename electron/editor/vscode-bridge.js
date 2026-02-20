(function () {
  // VS Code Webview bridge that emulates the Electron contextBridge API used by editor.js.
  const vscode = acquireVsCodeApi();

  let nextId = 1;
  const pending = new Map();

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || msg.kind !== 'response' || typeof msg.id !== 'number') return;
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error || 'Request failed'));
  });

  function request(method, params) {
    const id = nextId++;
    vscode.postMessage({ kind: 'request', id, method, params });
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
  }

  window.soundgenEditor = {
    openWavDialog: () => request('openWavDialog', {}),
    saveWavDialog: () => request('saveWavDialog', {}),

    pickClapPluginDialog: () => request('pickClapPluginDialog', {}),
    pickLv2BundleDialog: () => request('pickLv2BundleDialog', {}),
    clapListPlugins: (payload) => request('clapListPlugins', payload || {}),
    clapRenderPreview: (payload) => request('clapRenderPreview', payload || {}),

    editopsInit: (inPath) => request('editopsInit', { inPath: String(inPath || '') }),
    editopsInfo: (sessionId) => request('editopsInfo', { sessionId: String(sessionId || '') }),
    editopsOp: (payload) => request('editopsOp', payload || {}),
    editopsUndo: (sessionId) => request('editopsUndo', { sessionId: String(sessionId || '') }),
    editopsRedo: (sessionId) => request('editopsRedo', { sessionId: String(sessionId || '') }),
    editopsExport: (sessionId, outPath) => request('editopsExport', { sessionId: String(sessionId || ''), outPath: String(outPath || '') }),
    editopsClose: (sessionId) => request('editopsClose', { sessionId: String(sessionId || '') }),

    readFileBase64: (filePath) => request('readFileBase64', { filePath: String(filePath || '') })
  };
})();
