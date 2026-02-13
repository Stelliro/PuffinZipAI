/* ============================================================================
   Logger.js - Real-time Log Management
   ============================================================================ */

let logAutoScroll = true;
let logFilter = '';
let logRefreshInterval = 1000;
let logPollingTime = null;

function formatLogTime(timestamp) {
    try {
        const date = new DateTime(timestamp);
        return date.toLocaleTimeString();
    } catch (e) {
        // Fallback for ISO string
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

function addLogEntry(level, message, timestamp) {
    const logContainer = document.getElementById('log-container');
    if (!logContainer) return;
    
    // Create log entry element
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level.toLowerCase()}`;
    
    const time = formatLogTime(timestamp || new Date().toISOString());
    
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-level">${level}</span>
        <span class="log-message">${escapeHtml(message)}</span>
    `;
    
    logContainer.appendChild(entry);
    
    // Keep only last 500 entries for performance
    const entries = logContainer.querySelectorAll('.log-entry');
    if (entries.length > 500) {
        entries[0].remove();
    }
    
    // Auto-scroll if enabled
    if (logAutoScroll) {
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

function clearLogs() {
    const logContainer = document.getElementById('log-container');
    if (logContainer) {
        logContainer.innerHTML = '';
    }
}

function setLogFilter(filter) {
    logFilter = filter.toUpperCase();
    filterLogDisplay();
}

function filterLogDisplay() {
    const logContainer = document.getElementById('log-container');
    if (!logContainer) return;
    
    const entries = logContainer.querySelectorAll('.log-entry');
    entries.forEach(entry => {
        if (logFilter === '') {
            entry.style.display = '';
        } else {
            const level = entry.querySelector('.log-level').textContent;
            entry.style.display = level === logFilter ? '' : 'none';
        }
    });
}

function fetchAndDisplayLogs() {
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => {
                    addLogEntry(log.level, log.message, log.timestamp);
                });
            }
        })
        .catch(error => console.error('Error fetching logs:', error));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize logger controls
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart
    initializeChart();
    
    // Log filter
    const logFilter = document.getElementById('log-filter');
    if (logFilter) {
        logFilter.addEventListener('change', function(e) {
            setLogFilter(e.target.value);
        });
    }
    
    // Clear logs button
    const clearLogsBtn = document.getElementById('btn-clear-logs');
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', clearLogs);
    }
    
    // Auto-scroll checkbox
    const autoScrollCheckbox = document.getElementById('auto-scroll');
    if (autoScrollCheckbox) {
        autoScrollCheckbox.addEventListener('change', function(e) {
            logAutoScroll = e.target.checked;
            if (logAutoScroll) {
                // Scroll to bottom
                const logContainer = document.getElementById('log-container');
                if (logContainer) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            }
        });
    }
    
    // Refresh interval setting
    const refreshIntervalInput = document.getElementById('refresh-interval');
    if (refreshIntervalInput) {
        refreshIntervalInput.addEventListener('change', function(e) {
            logRefreshInterval = parseInt(e.target.value) || 1000;
            restartLogPolling();
        });
    }
    
    // Start polling logs
    startLogPolling();
});

function startLogPolling() {
    // Initial fetch
    fetchAndDisplayLogs();
    
    // Set up polling interval
    logPollingTime = setInterval(function() {
        fetch('/api/status')
            .then(r => r.json())
            .then(status => {
                // Update status indicator
                const statusIndicator = document.getElementById('status-indicator');
                const statusText = document.getElementById('status-text');
                
                if (statusIndicator) {
                    if (status.is_training) {
                        statusIndicator.classList.add('training');
                        statusIndicator.classList.remove('error');
                    } else {
                        statusIndicator.classList.remove('training');
                    }
                }
                
                if (statusText) {
                    statusText.textContent = status.is_training ? 'Training' : 'Ready';
                }
            });
        
        // Fetch new logs
        fetchAndDisplayLogs();
    }, logRefreshInterval);
}

function restartLogPolling() {
    if (logPollingTime) {
        clearInterval(logPollingTime);
    }
    startLogPolling();
}

// Export logs as JSON
function exportLogs() {
    const logContainer = document.getElementById('log-container');
    if (!logContainer) return;
    
    const entries = [];
    logContainer.querySelectorAll('.log-entry').forEach(entry => {
        const time = entry.querySelector('.log-time').textContent;
        const level = entry.querySelector('.log-level').textContent;
        const message = entry.querySelector('.log-message').textContent;
        entries.push({ time, level, message });
    });
    
    const data = JSON.stringify(entries, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    downloadBlob(blob, `logs_${new Date().toISOString()}.json`);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Wire up export button
document.addEventListener('DOMContentLoaded', function() {
    const exportBtn = document.getElementById('btn-export-logs');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportLogs);
    }
});
