
    
    /* ============================================================================
   App.js - Main Application Logic
   ============================================================================ */

class PuffinZipAIApp {
    constructor() {
        this.isTraining = false;
        this.currentGeneration = 0;
        this.updateInterval = null;
        this.init();
    }
    
    init() {
        this.restoreState();
        this.restoreMetricsFromLocalStorage();
        this.buildThemePreviewGrid();
        this.loadSavedTheme();
        this.setupEventListeners();
        this.loadCompressionMethods();
        // Polling slowed and delayed to prevent startup spam
        setTimeout(() => this.startStatusPolling(), 2000);
        this.initializeControlStates();
    }
    
    initializeControlStates() {
        // Initialize infinite generations checkbox state
        const infiniteCheckbox = document.getElementById('infinite-generations');
        const generationsInput = document.getElementById('generations');
        if (infiniteCheckbox && generationsInput) {
            generationsInput.disabled = infiniteCheckbox.checked;
        }
    }
    
    // ========================================================================
    // Tab Navigation
    // ========================================================================
    
    setupEventListeners() {
        // Tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.closest('.tab-btn')));
        });
        
        // Training controls
        document.getElementById('btn-start').addEventListener('click', () => this.startTraining());
        document.getElementById('btn-stop').addEventListener('click', () => this.stopTraining());
        document.getElementById('btn-reset').addEventListener('click', () => this.resetTraining());
        
        // Theme selector
        const themeSelector = document.getElementById('theme-selector');
        if (themeSelector) {
            themeSelector.addEventListener('change', (e) => this.setTheme(e.target.value));
        }
        
        // Settings
        document.getElementById('btn-export-metrics').addEventListener('click', () => this.exportMetrics());
        
        // Mutation rate slider
        document.getElementById('mutation-rate').addEventListener('input', (e) => {
            document.getElementById('mutation-rate-display').textContent = e.target.value + '%';
        });
        
        // Infinite generations checkbox
        const infiniteCheckbox = document.getElementById('infinite-generations');
        const generationsInput = document.getElementById('generations');
        if (infiniteCheckbox) {
            infiniteCheckbox.addEventListener('change', (e) => {
                generationsInput.disabled = e.target.checked;
            });
        }
    }
    
    switchTab(tabBtn) {
        // Remove active from all buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        
        // Remove active from all content
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active to clicked button
        tabBtn.classList.add('active');
        
        // Add active to corresponding content
        const tabName = tabBtn.dataset.tab;
        const tabContent = document.getElementById(tabName + '-tab');
        if (tabContent) {
            tabContent.classList.add('active');
        }
    }
    
    // ========================================================================
    // Training Control
    // ========================================================================
    
    async startTraining() {
        const generationsInput = document.getElementById('generations');
        const infiniteCheckbox = document.getElementById('infinite-generations');
        const stoppingCriteria = document.querySelector('input[name="stopping-criteria"]:checked').value;
        
        let generations = null;
        if (!infiniteCheckbox.checked) {
            generations = parseInt(generationsInput.value);
            if (isNaN(generations) || generations < 1) {
                alert('Please enter valid number of generations');
                return;
            }
        }
        
        const populationSize = parseInt(document.getElementById('population-size').value);
        
        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    generations,
                    population_size: populationSize,
                    infinite_generations: infiniteCheckbox.checked,
                    stopping_criteria: stoppingCriteria
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.isTraining = true;
                this.updateTrainingUI();
                this.startMetricsPolling();
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error starting training:', error);
            alert('Failed to start training: ' + error.message);
        }
    }
    
    async stopTraining() {
        try {
            const response = await fetch('/api/training/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.isTraining = false;
                this.updateTrainingUI();
                if (this.updateInterval) clearInterval(this.updateInterval);
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }
    
    async resetTraining() {
        if (confirm('Are you sure you want to reset all training data?')) {
            try {
                await fetch('/api/training/reset', {
                    method: 'POST'
                });
                
                // Clear chart
                if (fitnessChart) {
                    fitnessChart.data.labels = [];
                    fitnessChart.data.datasets[0].data = [];
                    fitnessChart.data.datasets[1].data = [];
                    fitnessChart.update();
                }
                
                // Reset values
                this.currentGeneration = 0;
                
                // Clear cached metrics from localStorage
                try {
                    localStorage.removeItem('trainingMetrics');
                    localStorage.removeItem('metricsTimestamp');
                } catch (e) {
                    console.warn('Could not clear localStorage:', e);
                }
                
                this.updateTrainingUI();
            } catch (error) {
                console.error('Error resetting training:', error);
            }
        }
    }
    
    updateTrainingUI() {
        const btnStart = document.getElementById('btn-start');
        const btnStop = document.getElementById('btn-stop');
        const statusVal = document.getElementById('training-status');
        
        if (this.isTraining) {
            btnStart.disabled = true;
            btnStop.disabled = false;
            statusVal.textContent = 'Training';
            statusVal.style.color = '#f59e0b';
        } else {
            btnStart.disabled = false;
            btnStop.disabled = true;
            statusVal.textContent = 'Idle';
            statusVal.style.color = '#10b981';
        }
    }
    
    startMetricsPolling() {
        this.updateInterval = setInterval(() => {
            this.updateMetrics();
            this.loadValidationResults();
        }, 500);
    }
    
    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            if (data.metrics.length > 0) {
                const lastMetric = data.metrics[data.metrics.length - 1];
                
                // Update UI
                document.getElementById('gen-count').textContent = lastMetric.generation;
                document.getElementById('metric-fitness').textContent = lastMetric.fitness.toFixed(4);
                document.getElementById('metric-compression').textContent = 
                    (lastMetric.compression_ratio * 100).toFixed(2) + '%';
                document.getElementById('metric-evo-time').textContent = 
                    lastMetric.evolution_time.toFixed(2) + 's';
                
                // Update chart
                addMetricToChart(lastMetric);
                
                // Save metrics to localStorage for persistence across page refreshes
                this.saveMetricsToLocalStorage(data.metrics);
            }
            
            // Update status
            const status = await fetch('/api/status').then(r => r.json());
            const wasTraining = this.isTraining;
            this.isTraining = status.is_training;
            
            if (!this.isTraining && this.updateInterval) {
                clearInterval(this.updateInterval);
                // Update UI to show Start button again when training stops
                if (wasTraining) {
                    this.updateTrainingUI();
                }
            }
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }
    
    saveMetricsToLocalStorage(metrics) {
        try {
            localStorage.setItem('trainingMetrics', JSON.stringify(metrics));
            localStorage.setItem('metricsTimestamp', new Date().toISOString());
        } catch (error) {
            console.warn('Could not save metrics to localStorage:', error);
        }
    }
    
    restoreMetricsFromLocalStorage() {
        try {
            const savedMetrics = localStorage.getItem('trainingMetrics');
            if (savedMetrics) {
                const metrics = JSON.parse(savedMetrics);
                
                // Update UI with last metric
                if (metrics.length > 0) {
                    const lastMetric = metrics[metrics.length - 1];
                    document.getElementById('gen-count').textContent = lastMetric.generation;
                    document.getElementById('metric-fitness').textContent = lastMetric.fitness.toFixed(4);
                    document.getElementById('metric-compression').textContent = 
                        (lastMetric.compression_ratio * 100).toFixed(2) + '%';
                    document.getElementById('metric-evo-time').textContent = 
                        lastMetric.evolution_time.toFixed(2) + 's';
                    
                    // Restore chart data
                    if (window.fitnessChart && window.fitnessChart.data) {
                        metrics.forEach(metric => {
                            if (!window.fitnessChart.data.labels.includes(metric.generation.toString())) {
                                addMetricToChart(metric);
                            }
                        });
                    }
                }
            }
        } catch (error) {
            console.warn('Could not restore metrics from localStorage:', error);
        }
    }
    
    async loadValidationResults() {
        try {
            const response = await fetch('/api/validation');
            const data = await response.json();
            
            const validationContainer = document.getElementById('validation-container');
            const validationPending = document.getElementById('validation-pending');
            const validationResults = document.getElementById('validation-results');
            
            if (data.success && data.validation) {
                // Validation has been completed
                validationPending.style.display = 'none';
                validationResults.style.display = 'block';
                
                // Set badge and title
                const badge = document.getElementById('validation-badge');
                const title = document.getElementById('validation-title');
                const subtitle = document.getElementById('validation-subtitle');
                
                if (data.validation.success) {
                    badge.className = 'validation-badge success';
                    title.textContent = '✓ AI Compression Success!';
                    subtitle.textContent = 'Your AI beat the baseline methods!';
                } else {
                    badge.className = 'validation-badge failure';
                    title.textContent = '✗ AI Did Not Outperform Baseline';
                    subtitle.textContent = 'AI compression is still learning...';
                }
                
                // Format sizes
                const formatSize = (bytes) => {
                    if (bytes === null) return '—';
                    return Math.round(bytes / 1024) + ' KB';
                };
                
                const formatPercent = (num) => {
                    if (num === null || num === undefined) return '—';
                    const percent = parseFloat(num);
                    if (isNaN(percent)) return percent;
                    return (percent > 0 ? '+' : '') + percent.toFixed(2) + '%';
                };
                
                // Update metrics
                document.getElementById('val-original-size').textContent = 
                    formatSize(data.validation.original_size);
                document.getElementById('val-ai-size').textContent = 
                    formatSize(data.validation.ai_compressed_size);
                document.getElementById('val-baseline-size').textContent = 
                    formatSize(data.validation.baseline_compressed_size);
                document.getElementById('val-baseline-method').textContent = 
                    data.validation.baseline_method || '—';
                document.getElementById('val-improvement').textContent = 
                    formatPercent(data.validation.ai_improvement_percent);
            } else {
                // Still waiting for validation
                validationPending.style.display = 'flex';
                validationResults.style.display = 'none';
            }
        } catch (error) {
            console.warn('Validation endpoint not yet available:', error);
        }
    }
    
    // ========================================================================
    // Compression Methods
    // ========================================================================
    
    async loadCompressionMethods() {
        try {
            const response = await fetch('/api/compression-methods');
            const data = await response.json();
            
            document.getElementById('total-methods').textContent = data.count;
            
            const novelCount = data.methods.filter(m => m.is_novelty).length;
            document.getElementById('novel-methods').textContent = novelCount;
            
            this.displayMethods(data.methods);
        } catch (error) {
            console.error('Error loading compression methods:', error);
        }
    }
    
    displayMethods(methods) {
        const grid = document.getElementById('methods-grid');
        
        if (methods.length === 0) {
            grid.innerHTML = '<div class="loading">No compression methods available</div>';
            return;
        }
        
        grid.innerHTML = methods.map(method => `
            <div class="method-card">
                <div class="method-name">
                    ${method.is_novelty ? '✨' : '📦'} ${escapeHtml(method.name)}
                </div>
                <div class="method-language">
                    Language: <strong>${method.language}</strong>
                </div>
                ${method.is_novelty ? '<span class="method-badge novel">Novel</span>' : ''}
                ${method.language === 'rust' ? '<span class="method-badge rust">Rust</span>' : ''}
                ${method.patterns.length > 0 ? `
                    <div class="method-patterns">
                        <strong>Patterns:</strong> ${method.patterns.join(', ')}
                    </div>
                ` : ''}
            </div>
        `).join('');
    }
    
    // ========================================================================
    // Theme Management
    // ========================================================================
    
    async loadThemes() {
        try {
            const response = await fetch('/api/themes');
            const data = await response.json();
            
            if (data.themes) {
                console.log('Themes loaded:', Object.keys(data.themes).length);
            }
        } catch (error) {
            console.error('Error loading themes:', error);
        }
    }
    
    setTheme(themeName) {
        try {
            // Get CSS class from theme name
            const themeClass = this.getThemeClass(themeName);
            
            // Remove all theme classes
            document.body.className = document.body.className
                .split(' ')
                .filter(cls => !cls.startsWith('theme-'))
                .join(' ');
            
            // Add new theme class
            if (themeClass) {
                document.body.classList.add(themeClass);
            }
            
            // Update grid selection
            this.updateThemeGridSelection(themeName);
            
            // Save preference
            localStorage.setItem('selectedTheme', themeName);
            
            console.log('Theme changed to:', themeName);
        } catch (error) {
            console.error('Error setting theme:', error);
        }
    }
    
    getThemeClass(themeName) {
        // Map theme names to CSS classes
        const themeMap = {
            'Nordic Dark (Default)': 'theme-nordic',
            'Dracula': 'theme-dracula',
            'Solarized Light': 'theme-solarized-light',
            'Monokai Pro': 'theme-monokai',
            'Oceanic Next': 'theme-oceanic',
            'GitHub Dark': 'theme-github-dark',
            'Zenburn': 'theme-zenburn',
            'Material Darker': 'theme-material',
            'Gruvbox Dark': 'theme-gruvbox',
            'Tomorrow Night Blue': 'theme-tomorrow-blue',
            'Forest Green': 'theme-forest-green',
            'Crimson Night': 'theme-crimson',
            'Electric Blue': 'theme-electric-blue',
            'Golden Sand': 'theme-golden-sand',
            'Neon Glow': 'theme-neon-glow',
            'Matrix Green': 'theme-matrix-green',
            'Sunset Orange': 'theme-sunset-orange',
            'Lavender Dream': 'theme-lavender',
            'Paper White': 'theme-paper-white',
            'Coffee House': 'theme-coffee'
        };
        
        return themeMap[themeName] || 'theme-nordic';
    }
    
    buildThemePreviewGrid() {
        const themes = [
            'Nordic Dark (Default)',
            'Dracula',
            'Solarized Light',
            'Monokai Pro',
            'Oceanic Next',
            'GitHub Dark',
            'Zenburn',
            'Material Darker',
            'Gruvbox Dark',
            'Tomorrow Night Blue',
            'Forest Green',
            'Crimson Night',
            'Electric Blue',
            'Golden Sand',
            'Neon Glow',
            'Matrix Green',
            'Sunset Orange',
            'Lavender Dream',
            'Paper White',
            'Coffee House'
        ];
        
        const grid = document.getElementById('theme-preview-grid');
        if (!grid) return;
        
        const themeColors = {
            'Nordic Dark (Default)': '#2E3440',
            'Dracula': '#282A36',
            'Solarized Light': '#FDF6E3',
            'Monokai Pro': '#2D2A2E',
            'Oceanic Next': '#1B2B34',
            'GitHub Dark': '#0D1117',
            'Zenburn': '#383838',
            'Material Darker': '#212121',
            'Gruvbox Dark': '#282828',
            'Tomorrow Night Blue': '#002451',
            'Forest Green': '#1B4D2A',
            'Crimson Night': '#4D1B2A',
            'Electric Blue': '#0A1F2E',
            'Golden Sand': '#4D3D1B',
            'Neon Glow': '#0F0F2E',
            'Matrix Green': '#0B2E1B',
            'Sunset Orange': '#4D2A1B',
            'Lavender Dream': '#3D2A4D',
            'Paper White': '#F5F5F5',
            'Coffee House': '#2B2B2B'
        };
        
        grid.innerHTML = '';
        
        themes.forEach(themeName => {
            const square = document.createElement('div');
            square.className = 'theme-preview-square';
            square.style.backgroundColor = themeColors[themeName] || '#333';
            square.textContent = themeName.split(' ').slice(0, 2).join('\n');
            square.title = themeName;
            square.dataset.theme = themeName;
            
            square.addEventListener('click', () => {
                this.setTheme(themeName);
                this.updateThemeGridSelection(themeName);
            });
            
            grid.appendChild(square);
        });
        
        // Set active theme in grid
        const savedTheme = localStorage.getItem('selectedTheme') || 'Nordic Dark (Default)';
        this.updateThemeGridSelection(savedTheme);
    }
    
    updateThemeGridSelection(themeName) {
        document.querySelectorAll('.theme-preview-square').forEach(square => {
            if (square.dataset.theme === themeName) {
                square.classList.add('active');
            } else {
                square.classList.remove('active');
            }
        });
    }
    
    loadSavedTheme() {
        const savedTheme = localStorage.getItem('selectedTheme') || 'Nordic Dark (Default)';
        this.setTheme(savedTheme);
    }
    
    startStatusPolling() {
        setInterval(() => {
            fetch('/api/status')
                .then(r => r.json())
                .then(status => {
                    this.isTraining = status.is_training;
                    
                    // Update status indicator
                    const indicator = document.getElementById('status-indicator');
                    if (indicator) {
                        if (status.is_training) {
                            indicator.classList.add('training');
                        } else {
                            indicator.classList.remove('training');
                        }
                    }
                })
                .catch(error => console.error('Status check error:', error));
        }, 1000);
    }
    
    // ========================================================================
    // Export Functions
    // ========================================================================
    
    async exportMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            const csv = this.convertToCSV(data.metrics);
            const blob = new Blob([csv], { type: 'text/csv' });
            this.downloadBlob(blob, `metrics_${new Date().toISOString()}.csv`);
        } catch (error) {
            alert('Error exporting metrics: ' + error.message);
        }
    }
    
    convertToCSV(metrics) {
        const headers = ['Generation', 'Fitness', 'Compression Ratio', 'Evolution Time', 'Timestamp'];
        const rows = metrics.map(m => [
            m.generation,
            m.fitness.toFixed(4),
            (m.compression_ratio * 100).toFixed(2) + '%',
            m.evolution_time.toFixed(2),
            m.timestamp
        ]);
        
        return [headers, ...rows]
            .map(row => row.map(cell => `"${cell}"`).join(','))
            .join('\n');
    }
    
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    // ========================================================================
    // Theme
    // ========================================================================
    
    toggleDarkMode(enabled) {
        if (enabled) {
            document.body.classList.add('dark-mode');
            document.body.classList.remove('light-mode');
        } else {
            document.body.classList.remove('dark-mode');
            document.body.classList.add('light-mode');
        }
        localStorage.setItem('darkMode', enabled);
    }
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    restoreState() {
        // Load theme preference
        const savedTheme = localStorage.getItem('selectedTheme') || 'Nordic Dark (Default)';
        this.setTheme(savedTheme);
        document.getElementById('theme-selector').value = savedTheme;
        
        // Load other preferences
        const refreshInterval = localStorage.getItem('refreshInterval') || '1000';
        document.getElementById('refresh-interval').value = refreshInterval;
        
        // Check if training was in progress, and resume polling if so
        // this.checkAndResumeTraining(); // Disabled to stop auto-firing on page load
    }
    
    async checkAndResumeTraining() {
        try {
            const status = await fetch('/api/status').then(r => r.json());
            if (status.is_training) {
                // Training is still running on the server, resume client polling
                this.isTraining = true;
                this.updateTrainingUI();
                this.startMetricsPolling();
                console.log('Resumed training session from existing server process');
            }
        } catch (error) {
            console.warn('Could not check training status on init:', error);
        }
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', function() {
    const app = new PuffinZipAIApp();
    
    // Update navbar with server info
    const host = window.location.hostname;
    const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    document.getElementById('server-host').textContent = `${host}:${port}`;
    document.getElementById('last-update').textContent = new Date().toLocaleString();
});
