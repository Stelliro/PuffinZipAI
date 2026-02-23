/* ============================================================================
   Logger.js - Log Management (Clean)
   ============================================================================ */

function getLogTimestamp() {
    const now = new Date();
    return '[' + now.toLocaleTimeString('en-US', { hour12: false }) + ']';
}

function logMessage(message, level = 'INFO') {
    const container = document.getElementById('log-container');
    if (!container) return;

    const div = document.createElement('div');
    // Map log levels to simple classes
    const levelClass = level.toLowerCase() === 'error' ? 'log-error' : 'log-info';
    div.className = `log-entry ${levelClass}`;
    
    let color = '#8be9fd'; // Cyan (Info)
    if (level === 'ERROR') color = '#ff5555'; // Red
    if (level === 'WARNING') color = '#f1fa8c'; // Yellow
    if (level === 'SUCCESS') color = '#50fa7b'; // Green

    div.innerHTML = `
        <span class="log-time" style="color:#6272a4; margin-right:8px;">${getLogTimestamp()}</span>
        <span class="log-level" style="color:${color}; font-weight:bold; margin-right:8px;">[${level}]</span>
        <span class="log-msg" style="color:#f8f8f2;">${escapeHtml(message)}</span>
    `;

    container.appendChild(div);

    const autoScroll = document.getElementById('auto-scroll');
    if (!autoScroll || autoScroll.checked) {
        container.scrollTop = container.scrollHeight;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

console.log("Logger loaded.");