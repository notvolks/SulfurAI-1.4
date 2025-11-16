
const { app, BrowserWindow } = require('electron');

app.on('ready', () => {
    let win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: { nodeIntegration: true }
    });
    win.loadURL('http://127.0.0.1:5000/dashboard/');
});
