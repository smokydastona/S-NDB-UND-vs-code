const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('soundgenEditor', {
  openWavDialog: () => ipcRenderer.invoke('editor:openWavDialog'),
  saveWavDialog: () => ipcRenderer.invoke('editor:saveWavDialog'),

  pickClapPluginDialog: () => ipcRenderer.invoke('editor:pickClapPluginDialog'),
  pickLv2BundleDialog: () => ipcRenderer.invoke('editor:pickLv2BundleDialog'),
  clapListPlugins: (payload) => ipcRenderer.invoke('editor:clapListPlugins', payload || {}),
  clapRenderPreview: (payload) => ipcRenderer.invoke('editor:clapRenderPreview', payload || {}),

  editopsInit: (inPath) => ipcRenderer.invoke('editor:editopsInit', String(inPath || '')),
  editopsInfo: (sessionId) => ipcRenderer.invoke('editor:editopsInfo', String(sessionId || '')),
  editopsOp: (payload) => ipcRenderer.invoke('editor:editopsOp', payload || {}),
  editopsUndo: (sessionId) => ipcRenderer.invoke('editor:editopsUndo', String(sessionId || '')),
  editopsRedo: (sessionId) => ipcRenderer.invoke('editor:editopsRedo', String(sessionId || '')),
  editopsExport: (sessionId, outPath) => ipcRenderer.invoke('editor:editopsExport', { sessionId: String(sessionId || ''), outPath: String(outPath || '') }),
  editopsClose: (sessionId) => ipcRenderer.invoke('editor:editopsClose', String(sessionId || '')),

  readFileBase64: (filePath) => ipcRenderer.invoke('editor:readFileBase64', String(filePath || '')),
});
