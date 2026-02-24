/* ============================================================================
   PuffinZipAI Web UI — Main Application
   ============================================================================ */

class PuffinZipAIApp {
    constructor() {
        this.isTraining = false;
        this.canContinue = false;
        this.chart = null;
        this.popPage = 1;
        this.popTotalPages = 1;
        this.popExpandedGens = new Set();  // Track which generations are expanded
        this.init();
    }

    init() {
        console.log("🚀 PuffinZipAI App Initializing...");
        this.initChart();
        this.setupEventListeners();
        this.startLoops();
        this.loadCompressionMethods();
        this.loadSystemDefaults();
        this.loadHardwareProfile();
    }

    /** Fetch system limits and pre-fill form defaults */
    async loadSystemDefaults() {
        try {
            const res = await fetch('/api/status');
            const d = await res.json();
            const lim = d.system_limits || {};
            const wEl = document.getElementById('cpu-workers');
            if (wEl && lim.default_workers) {
                wEl.value = lim.default_workers;
            }
        } catch (_) {}
    }

    /** Fetch hardware profile + run presets and wire up preset buttons */
    async loadHardwareProfile() {
        try {
            const res = await fetch('/api/hardware-profile');
            const d = await res.json();
            this._presets = d.presets || {};
            const hw = d.hardware || {};

            // Show hardware info chips
            const hwRow = document.getElementById('hw-info-row');
            if (hwRow) {
                const gpuChip = document.getElementById('hw-chip-gpu');
                const ramChip = document.getElementById('hw-chip-ram');
                const cpuChip = document.getElementById('hw-chip-cpu');
                if (gpuChip) gpuChip.textContent = hw.has_gpu
                    ? `🖥 ${hw.gpu_count}× ${hw.gpu_name} (${hw.gpu_vram_gb}GB)`
                    : '🖥 CPU only';
                if (ramChip) ramChip.textContent = `💾 ${hw.ram_gb} GB RAM`;
                if (cpuChip) cpuChip.textContent = `⚙ ${hw.cpu_cores} cores`;
                hwRow.style.display = 'flex';
            }

            // Wire preset buttons
            const btnTest = document.getElementById('preset-test');
            const btnMed = document.getElementById('preset-medium');
            const btnMax = document.getElementById('preset-max');

            const applyPreset = (key, btn) => {
                const preset = this._presets[key];
                if (!preset) return;
                // Remove active class from all preset buttons
                document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                // Apply values
                const q = id => document.getElementById(id);
                if (q('generations')) q('generations').value = preset.num_generations;
                if (q('population-size')) q('population-size').value = preset.population_size;
                if (q('batch-size')) q('batch-size').value = preset.batch_size;
                if (q('cpu-workers')) q('cpu-workers').value = preset.cpu_workers;
                if (q('device-select')) q('device-select').value = preset.target_device;
                if (q('infinite-generations')) q('infinite-generations').checked = !!preset.infinite;
                // Show description
                const desc = document.getElementById('preset-description');
                if (desc) desc.textContent = preset.description || '';
            };

            if (btnTest) btnTest.addEventListener('click', () => applyPreset('test', btnTest));
            if (btnMed)  btnMed.addEventListener('click',  () => applyPreset('medium', btnMed));
            if (btnMax)  btnMax.addEventListener('click',  () => applyPreset('max', btnMax));

            // Clear active preset state when user manually changes any config field
            const configInputs = ['generations', 'population-size', 'batch-size', 'cpu-workers', 'device-select', 'infinite-generations'];
            configInputs.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.addEventListener('input', () => {
                    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
                    const desc = document.getElementById('preset-description');
                    if (desc) desc.textContent = '';
                });
            });

        } catch (e) {
            console.warn('Failed to load hardware profile:', e);
        }
    }

     /* ------------------------------------------------------------------
         Chart — 3 datasets: Compression Rate, Benchmark Size, Complexity Value
         ------------------------------------------------------------------ */
    initChart() {
        const ctx = document.getElementById('fitnessChart');
        if (!ctx) return;

        const existing = Chart.getChart(ctx);
        if (existing) existing.destroy();

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Compression Rate (%)',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74,222,128,0.1)',
                        tension: 0.3, borderWidth: 2, pointRadius: 0,
                        fill: true, yAxisID: 'y'
                    },
                    {
                        label: 'Benchmark Size (KB)',
                        data: [],
                        borderColor: '#8be9fd',
                        backgroundColor: 'rgba(139,233,253,0.05)',
                        tension: 0.3, borderWidth: 2, pointRadius: 0,
                        fill: false, borderDash: [5,5], yAxisID: 'y2'
                    },
                    {
                        label: 'Complexity Value',
                        data: [],
                        borderColor: '#fbbf24',
                        backgroundColor: 'rgba(251,191,36,0.08)',
                        tension: 0.3, borderWidth: 1.5, pointRadius: 0,
                        fill: true, yAxisID: 'y3'
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { display: true, labels: { color: '#f8f8f2', boxWidth: 14, font: { size: 11 } } }
                },
                scales: {
                    x: { grid: { color: '#333' }, ticks: { color: '#f8f8f2', maxTicksLimit: 20 } },
                    y: {
                        grid: { color: '#333' }, position: 'left',
                        suggestedMin: 90,
                        ticks: { color: '#4ade80' },
                        title: { display: true, text: 'Compression Score (%)', color: '#4ade80' }
                    },
                    y2: {
                        grid: { drawOnChartArea: false }, beginAtZero: true, position: 'right',
                        ticks: { color: '#8be9fd' },
                        title: { display: true, text: 'Benchmark (KB)', color: '#8be9fd' }
                    },
                    y3: {
                        grid: { drawOnChartArea: false },
                        beginAtZero: true,
                        suggestedMax: 100,
                        position: 'right',
                        offset: true,
                        ticks: { color: '#fbbf24', stepSize: 25 },
                        title: { display: true, text: 'Complexity (1-100)', color: '#fbbf24' }
                    }
                }
            }
        });

        window.fitnessChart = this.chart;
    }

    /* ------------------------------------------------------------------
       Event Listeners
       ------------------------------------------------------------------ */
    setupEventListeners() {
        const q = id => document.getElementById(id);

        q('btn-start')?.addEventListener('click', () => this.startTraining());
        q('btn-stop')?.addEventListener('click', () => this.stopTraining());
        q('btn-continue')?.addEventListener('click', () => this.continueTraining());
        q('btn-reset')?.addEventListener('click', () => { window.resetChart?.(); });
        q('btn-save-checkpoint')?.addEventListener('click', () => this.saveCheckpoint());
        q('btn-view-checkpoints')?.addEventListener('click', () => this.viewCheckpoints());
        q('modal-close')?.addEventListener('click', () => {
            q('checkpoints-modal').style.display = 'none';
        });

        // Population history pagination
        q('pop-page-prev')?.addEventListener('click', () => {
            if (this.popPage > 1) { this.popPage--; this.pollPopulation(); }
        });
        q('pop-page-next')?.addEventListener('click', () => {
            if (this.popPage < this.popTotalPages) { this.popPage++; this.pollPopulation(); }
        });
        q('pop-expand-all')?.addEventListener('click', () => {
            document.querySelectorAll('.gen-agents-table').forEach(t => t.style.display = '');
            document.querySelectorAll('.gen-toggle').forEach(b => { b.textContent = '▾'; });
            // Track all as expanded
            document.querySelectorAll('.gen-header').forEach(h => {
                const g = parseInt(h.dataset.gen);
                if (!isNaN(g)) this.popExpandedGens.add(g);
            });
        });
        q('pop-collapse-all')?.addEventListener('click', () => {
            document.querySelectorAll('.gen-agents-table').forEach(t => t.style.display = 'none');
            document.querySelectorAll('.gen-toggle').forEach(b => { b.textContent = '▸'; });
            this.popExpandedGens.clear();
        });
        q('btn-clear-logs')?.addEventListener('click', () => {
            const c = q('log-container');
            if (c) c.innerHTML = '';
        });
        q('btn-export-logs')?.addEventListener('click', () => this.exportLogs());
        q('btn-export-metrics')?.addEventListener('click', () => this.exportMetrics());

        // Mutation rate slider display
        q('mutation-rate')?.addEventListener('input', e => {
            const d = q('mutation-rate-display');
            if (d) d.textContent = e.target.value + '%';
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => {
                    c.classList.remove('active');
                    c.style.display = 'none';
                });
                btn.classList.add('active');
                const content = document.getElementById(`${btn.dataset.tab}-tab`);
                if (content) { content.classList.add('active'); content.style.display = 'block'; }
            });
        });

        // Top-level tab switching (Dashboard / Deep Dive)
        document.querySelectorAll('.top-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.top-tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.top-tab-content').forEach(c => {
                    c.classList.remove('active');
                    c.style.display = 'none';
                });
                btn.classList.add('active');
                const content = document.getElementById(`${btn.dataset.topTab}-top-tab`);
                if (content) { content.classList.add('active'); content.style.display = 'block'; }
            });
        });
    }

    /* ------------------------------------------------------------------
       Training Start / Stop
       ------------------------------------------------------------------ */
    async startTraining() {
        const btnStart = document.getElementById('btn-start');
        if (btnStart) btnStart.disabled = true;

        // If there's existing chart data from a previous run, archive and reset
        if (this.chart && this.chart.data.labels.length > 0) {
            window.archiveAndResetChart?.();
        }

        const gens = document.getElementById('generations')?.value || 1000;
        const pop = document.getElementById('population-size')?.value || 500;
        const batch = document.getElementById('batch-size')?.value || 10;
        const infinite = document.getElementById('infinite-generations')?.checked || false;
        const device = document.getElementById('device-select')?.value || 'GPU_AUTO';
        const workers = document.getElementById('cpu-workers')?.value || 4;

        try {
            await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_generations: parseInt(gens),
                    population_size: parseInt(pop),
                    batch_size: parseInt(batch),
                    infinite,
                    target_device: device,
                    cpu_workers: parseInt(workers)
                })
            });
            this.isTraining = true;
            this.popPage = 1;  // Reset page on new run
            this.popExpandedGens.clear();
            this.updateUIState();
        } catch (e) {
            alert("Failed to start: " + e.message);
            if (btnStart) btnStart.disabled = false;
        }
    }

    async stopTraining() {
        try {
            await fetch('/api/training/stop', { method: 'POST' });
            this.isTraining = false;
            this.updateUIState();
        } catch (e) { console.error(e); }
    }

    async continueTraining() {
        const btnContinue = document.getElementById('btn-continue');
        if (btnContinue) btnContinue.disabled = true;

        const gens = document.getElementById('generations')?.value || 100;
        const infinite = document.getElementById('infinite-generations')?.checked || false;

        try {
            const res = await fetch('/api/training/continue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    extra_generations: parseInt(gens),
                    infinite
                })
            });
            const d = await res.json();
            if (d.success) {
                this.isTraining = true;
                this.updateUIState();
            } else {
                alert('Continue failed: ' + (d.error || 'Unknown error'));
                if (btnContinue) btnContinue.disabled = false;
            }
        } catch (e) {
            alert('Continue request failed: ' + e.message);
            if (btnContinue) btnContinue.disabled = false;
        }
    }

    updateUIState() {
        const q = id => document.getElementById(id);
        const btnContinue = q('btn-continue');
        if (this.isTraining) {
            if (q('btn-start')) q('btn-start').disabled = true;
            if (q('btn-stop'))  q('btn-stop').disabled = false;
            if (btnContinue) btnContinue.disabled = true;
            if (q('training-status')) { q('training-status').innerText = 'Training'; q('training-status').style.color = '#4ade80'; }
            if (q('status-indicator')) q('status-indicator').className = 'status-indicator training';
            if (q('status-text')) q('status-text').innerText = 'Training';
        } else {
            if (q('btn-start')) q('btn-start').disabled = false;
            if (q('btn-stop'))  q('btn-stop').disabled = true;
            // Show Continue only when a completed run exists
            if (btnContinue) btnContinue.disabled = !this.canContinue;
            if (q('training-status')) { q('training-status').innerText = 'Idle'; q('training-status').style.color = ''; }
            if (q('status-indicator')) q('status-indicator').className = 'status-indicator';
            if (q('status-text')) q('status-text').innerText = this.canContinue ? 'Completed' : 'Ready';
        }
    }

    /* ------------------------------------------------------------------
       Polling Loops
       ------------------------------------------------------------------ */
    startLoops() {
        setInterval(() => this.pollStatus(), 1000);
        setInterval(async () => {
            await this.pollLogs();
            if (this.isTraining) await this.pollMetrics();
        }, 1000);
        // Population history: poll every 3s, only when tab is active
        setInterval(async () => {
            const pt = document.getElementById('population-tab');
            if (pt && (pt.classList.contains('active') || pt.style.display === 'block')) {
                // During training, auto-follow the latest page
                if (this.isTraining) this.popPage = Math.max(1, this.popTotalPages);
                await this.pollPopulation();
            }
        }, 3000);
    }

    async pollStatus() {
        try {
            const res = await fetch('/api/status');
            const d = await res.json();
            const changed = this.isTraining !== d.is_training || this.canContinue !== !!d.can_continue;
            this.isTraining = d.is_training;
            this.canContinue = !!d.can_continue;
            if (changed) this.updateUIState();

            const genEl = document.getElementById('gen-count');
            if (genEl) genEl.innerText = d.current_generation;

            const elapsed = d.evolution_time || 0;
            const evoEl = document.getElementById('metric-evo-time');
            if (evoEl) {
                if (elapsed < 60) evoEl.innerText = elapsed.toFixed(1) + 's';
                else if (elapsed < 3600) evoEl.innerText = (elapsed / 60).toFixed(1) + 'm';
                else evoEl.innerText = (elapsed / 3600).toFixed(2) + 'h';
            }
        } catch (_) {}
    }

    async pollMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const d = await res.json();

            /* --- KPI updates --- */
            const q = id => document.getElementById(id);
            const set = (id, v) => { const el = q(id); if (el) el.innerText = v; };

            // Convert savings-based ratio (0-100%) to compression score
            // (original / compressed * 100) which CAN exceed 100%.
            // 100% = no compression, 200% = 2:1, 300% = 3:1, etc.
            const savingsRatio = d.compression_ratio || 0;
            const compScore = (savingsRatio < 99.9)
                ? (10000 / (100 - savingsRatio))
                : 9999;
            set('score-value', compScore.toFixed(1) + '%');
            set('metric-file-size', d.benchmark_size || '0.00 MB');

            /* --- Complexity sidebar + KPI --- */
            const tier = d.complexity_tier || 'VERY_SIMPLE';
            const cval = d.complexity_value !== undefined ? d.complexity_value : 0;
            const budgetMB = d.tier_budget_mb || 5;
            const ceilingKB = d.tier_ceiling_kb || 256;

            set('metric-complexity-tier', tier.replace('_', ' '));
            // Complexity is already on a 1-100 scale from the server
            set('metric-complexity-level', `${cval} / 100`);
            set('metric-tier-info', `Budget: ${budgetMB} MB | Ceiling: ${ceilingKB} KB`);

            // Sidebar detail rows
            set('detail-pattern-tier', tier.replace('_', ' '));
            set('detail-tier-budget', budgetMB + ' MB');
            set('detail-tier-ceiling', ceilingKB + ' KB');
            // Compression targets per tier — ratio gate to advance to NEXT tier
            // Matches COMPLEXITY_RATIO_GATES in benchmark_evaluator.py
            const targets = {
                'VERY_SIMPLE': '≥ 25% to advance',
                'SIMPLE': '≥ 45% to advance',
                'MODERATE': '≥ 60% to advance',
                'COMPLEX': '≥ 70% to advance',
                'VERY_COMPLEX': 'Mastered ✓'
            };
            set('detail-compression-target', targets[tier] || '—');

            // Complexity progress bar (1-100 scale from server)
            const pct = Math.min(100, Math.max(1, cval));
            const bar = q('complexity-bar');
            if (bar) {
                bar.style.width = pct + '%';
                const lbl = q('complexity-bar-label');
                if (lbl) lbl.innerText = tier.replace('_', ' ');
            }

            /* --- Robustness / Anti-Corruption KPI + sidebar --- */
            const robustness = d.best_robustness || 0;
            const trainingPhase = d.training_phase || '—';
            const corruptionLvl = d.corruption_level || 0;
            const decompMM = d.decomp_mismatches || 0;
            const itemsEval = d.items_evaluated || 0;
            const successComp = d.successful_compressions || 0;

            // KPI card
            set('metric-robustness', robustness.toFixed(4));
            // Show short phase label in KPI sub
            set('metric-training-phase', trainingPhase || '—');

            // Sidebar detail rows
            set('detail-training-phase', trainingPhase || '—');
            set('detail-corruption-level', corruptionLvl.toFixed(3));
            set('detail-robustness', robustness.toFixed(4));
            set('detail-items-evaluated', itemsEval);
            set('detail-successful-comp', successComp);
            set('detail-decomp-mismatches', decompMM);

            /* --- Method Stats (which novel methods are working) --- */
            const methodStats = d.method_stats || {};
            const novelPipeline = d.novel_pipeline || 'none';
            set('detail-novel-pipeline', novelPipeline.replace(/_/g, ' '));

            const methodStatsEl = q('method-stats-grid');
            if (methodStatsEl) {
                const bytesS = methodStats.bytes_saved || {};
                const attempts = methodStats.attempts || {};
                const successes = methodStats.successes || {};
                const methods = ['RLE', 'AdvancedRLE', 'NovelMethod', 'ReferenceMethod'];
                let html = '';
                for (const m of methods) {
                    const att = attempts[m] || 0;
                    const suc = successes[m] || 0;
                    const saved = bytesS[m] || 0;
                    if (att === 0) continue;
                    const sucRate = att > 0 ? ((suc / att) * 100).toFixed(0) : '0';
                    const savedKB = (saved / 1024).toFixed(1);
                    const cls = saved > 0 ? 'method-positive' : (saved < 0 ? 'method-negative' : 'method-neutral');
                    const label = m === 'NovelMethod' ? `Novel (${novelPipeline.replace(/_/g, ' ')})` : m.replace(/([A-Z])/g, ' $1').trim();
                    html += `<div class="method-stat-row ${cls}">
                        <span class="method-label">${label}</span>
                        <span class="method-value">${savedKB} KB saved</span>
                        <span class="method-rate">${sucRate}% (${suc}/${att})</span>
                    </div>`;
                }
                methodStatsEl.innerHTML = html || '<div class="method-stat-row method-neutral">No method data yet</div>';
            }

            /* --- Chart update --- */
            if (this.chart && d.metrics && d.metrics.length > 0) {
                this.chart.data.labels = d.metrics.map(m => m.generation);
                this.chart.data.datasets[0].data = d.metrics.map(m => {
                    const ratio = m.ratio !== undefined ? m.ratio : 0;
                    // Convert savings % to compression score (> 100% = actual compression)
                    return ratio < 99.9 ? parseFloat((10000 / (100 - ratio)).toFixed(1)) : 9999;
                });
                this.chart.data.datasets[1].data = d.metrics.map(m => {
                    const bs = m.benchmark_size || 0;
                    return parseFloat((bs / 1024).toFixed(1));
                });
                // Complexity value per generation (if tracked)
                this.chart.data.datasets[2].data = d.metrics.map(m => m.complexity_value !== undefined ? m.complexity_value : 0);
                this.chart.update('none');
            }
        } catch (_) {}
    }

    async pollPopulation() {
        try {
            const res = await fetch(`/api/population/history?page=${this.popPage}&per_page=20`);
            const d = await res.json();
            const container = document.getElementById('population-history-container');
            if (!container) return;

            this.popTotalPages = d.total_pages || 1;
            const pageInfo = document.getElementById('pop-page-info');
            if (pageInfo) pageInfo.textContent = `Page ${d.page || 1} / ${this.popTotalPages}`;

            // Collect all generations to render (snapshots + live)
            const allGens = [];
            if (d.generations && d.generations.length > 0) {
                allGens.push(...d.generations);
            }
            if (d.live_generation && d.page === d.total_pages) {
                allGens.push(d.live_generation);
            }

            if (allGens.length === 0) {
                container.innerHTML = '<div class="placeholder-cell" style="padding:24px;text-align:center;">Waiting for data...</div>';
                return;
            }

            // Build HTML — each generation is a collapsible section
            let html = '';
            for (const gen of allGens) {
                const genNum = gen.generation || 0;
                const isLive = gen.is_live || false;
                const isExpanded = this.popExpandedGens.has(genNum) || isLive;
                const toggleChar = isExpanded ? '▾' : '▸';
                const displayStyle = isExpanded ? '' : 'display:none;';
                const liveTag = isLive ? ' <span class="live-badge">LIVE</span>' : '';
                const bestFit = gen.best_fitness !== undefined ? parseFloat(gen.best_fitness).toFixed(4) : '—';
                const avgFit = gen.avg_fitness !== undefined ? parseFloat(gen.avg_fitness).toFixed(4) : '—';
                const agentCount = gen.agent_count || 0;

                html += `<div class="gen-section">
                    <div class="gen-header" data-gen="${genNum}" onclick="window._pzApp.toggleGen(${genNum}, this)">
                        <span class="gen-toggle">${toggleChar}</span>
                        <span class="gen-title">Generation ${genNum}${liveTag}</span>
                        <span class="gen-meta">Agents: ${agentCount} | Best: ${bestFit} | Avg: ${avgFit}</span>
                    </div>
                    <div class="gen-agents-table" id="gen-table-${genNum}" style="${displayStyle}">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Agent ID</th>
                                    <th>Fitness</th>
                                    <th>Robustness</th>
                                    <th>Decomp</th>
                                    <th>Gen Born</th>
                                    <th>Thresholds</th>
                                </tr>
                            </thead>
                            <tbody>`;

                // Flatten agents from batches
                const agents = [];
                if (gen.batches) {
                    for (const batch of gen.batches) {
                        if (batch.agents) agents.push(...batch.agents);
                    }
                }

                if (agents.length === 0) {
                    html += '<tr><td colspan="6" class="placeholder-cell">No agent data</td></tr>';
                } else {
                    for (const a of agents) {
                        const fit = (a.fitness !== null && a.fitness !== undefined && a.fitness > -999)
                            ? parseFloat(a.fitness).toFixed(4) : 'Pending...';
                        const id = a.agent_id || a.id || 'Unknown';
                        const genBorn = a.generation_born !== undefined ? a.generation_born : (a.gen_born || 0);
                        const thresh = a.thresholds_str || a.thresholds || 'N/A';

                        // Robustness & decompression from evaluation_stats
                        const es = a.evaluation_stats || {};
                        const rFit = es.robustness_fitness;
                        const robCell = (rFit !== null && rFit !== undefined)
                            ? parseFloat(rFit).toFixed(4) : '—';
                        const decompMM = es.decomp_failures_mismatch;
                        const itemsEv = es.items_evaluated;
                        let decompCell = '—';
                        if (itemsEv !== undefined && itemsEv > 0) {
                            const mm = decompMM || 0;
                            const ok = itemsEv - mm;
                            decompCell = `${ok}/${itemsEv}`;
                            if (mm > 0) decompCell += ` <span style="color:#f87171;">(${mm} fail)</span>`;
                        }

                        html += `<tr>
                            <td style="color:#4ade80;">${id}</td>
                            <td>${fit}</td>
                            <td style="color:#c084fc;">${robCell}</td>
                            <td style="font-size:.9em;">${decompCell}</td>
                            <td>${genBorn}</td>
                            <td style="font-family:monospace;font-size:.9em;color:var(--text-muted);">${thresh}</td>
                        </tr>`;
                    }
                }

                html += `</tbody></table></div></div>`;
            }

            container.innerHTML = html;
        } catch (e) { console.warn("Pop history poll error", e); }
    }

    toggleGen(genNum, headerEl) {
        const table = document.getElementById(`gen-table-${genNum}`);
        const toggle = headerEl?.querySelector('.gen-toggle');
        if (!table) return;
        const wasHidden = table.style.display === 'none';
        table.style.display = wasHidden ? '' : 'none';
        if (toggle) toggle.textContent = wasHidden ? '▾' : '▸';
        if (wasHidden) {
            this.popExpandedGens.add(genNum);
        } else {
            this.popExpandedGens.delete(genNum);
        }
    }

    async pollLogs() {
        try {
            const res = await fetch('/api/logs');
            const logs = await res.json();
            const container = document.getElementById('log-container');
            if (!container || logs.length === 0) return;

            const filterEl = document.getElementById('log-filter');
            const filter = filterEl ? filterEl.value : '';

            logs.forEach(log => {
                if (filter && log.level !== filter) return;
                const div = document.createElement('div');
                div.className = `log-entry log-${log.level.toLowerCase()}`;
                let color = '#ccc';
                if (log.level === 'ERROR')   color = '#ef4444';
                if (log.level === 'WARNING') color = '#f59e0b';
                if (log.level === 'SUCCESS') color = '#4ade80';
                div.innerHTML = `<span class="log-level" style="color:${color};">[${log.level}]</span> <span class="log-message">${log.message}</span>`;
                container.appendChild(div);
            });

            while (container.children.length > 300) container.removeChild(container.firstChild);

            if (document.getElementById('auto-scroll')?.checked) {
                container.scrollTop = container.scrollHeight;
            }
        } catch (_) {}
    }

    /* ------------------------------------------------------------------
       Checkpoints
       ------------------------------------------------------------------ */
    async saveCheckpoint() {
        const nameInput = document.getElementById('checkpoint-name');
        const name = nameInput ? nameInput.value.trim() : '';
        const btn = document.getElementById('btn-save-checkpoint');
        if (btn) btn.disabled = true;
        try {
            const res = await fetch('/api/checkpoint/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const data = await res.json();
            if (data.success) {
                alert(`Checkpoint '${data.name}' saved!`);
                if (nameInput) nameInput.value = '';
            } else {
                alert('Save failed: ' + (data.error || 'Unknown'));
            }
        } catch (e) { alert('Save failed: ' + e.message); }
        finally { if (btn) btn.disabled = false; }
    }

    async viewCheckpoints() {
        const modal = document.getElementById('checkpoints-modal');
        const list = document.getElementById('checkpoints-list');
        if (!modal || !list) return;
        modal.style.display = 'flex';
        list.innerHTML = '<p>Loading...</p>';
        try {
            const res = await fetch('/api/checkpoint/list');
            const data = await res.json();
            if (data.checkpoints && data.checkpoints.length > 0) {
                list.innerHTML = data.checkpoints.map(cp => {
                    const name = cp.name || cp.checkpoint_name || 'Unnamed';
                    const key = cp.key || name;
                    const gen = cp.generation || cp.total_generations_elapsed || '?';
                    const fitness = cp.best_fitness !== undefined ? parseFloat(cp.best_fitness).toFixed(4) : '?';
                    const date = cp.timestamp || cp.date || '';
                    return `<div class="checkpoint-item" style="display:flex;align-items:center;justify-content:space-between;">
                        <div>
                            <strong style="color:#4ade80;">${name}</strong>
                            <span style="color:var(--text-muted);margin-left:12px;">Gen ${gen} | Fitness: ${fitness}</span>
                            <span style="color:var(--text-muted);margin-left:12px;font-size:.85em;">${date}</span>
                        </div>
                        <button class="btn btn-sm" style="color:var(--danger-color);background:transparent;border:1px solid var(--danger-color);cursor:pointer;font-size:.8em;padding:2px 8px;"
                            onclick="window._pzApp._deleteCheckpoint('${key.replace(/'/g, "\\'")}')">🗑 Delete</button>
                    </div>`;
                }).join('');
            } else {
                list.innerHTML = '<p style="color:var(--text-muted);">No checkpoints saved yet</p>';
            }
        } catch (e) {
            list.innerHTML = `<p style="color:var(--danger-color);">Error: ${e.message}</p>`;
        }
    }

    async _deleteCheckpoint(key) {
        if (!confirm(`Delete checkpoint "${key}"?`)) return;
        try {
            const res = await fetch('/api/checkpoint/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key })
            });
            const data = await res.json();
            if (data.success) {
                this.viewCheckpoints();  // refresh the list
            } else {
                alert('Delete failed: ' + (data.error || 'Unknown'));
            }
        } catch (e) { alert('Delete failed: ' + e.message); }
    }

    /* ------------------------------------------------------------------
       Compression Methods
       ------------------------------------------------------------------ */
    async loadCompressionMethods() {
        try {
            const res = await fetch('/api/compression-methods');
            const methods = await res.json();
            const grid = document.getElementById('methods-grid');
            if (grid && methods.length) {
                grid.innerHTML = methods.map(m => `<div class="method-card">${m}</div>`).join('');
                const el = document.getElementById('total-methods');
                if (el) el.innerText = methods.length;
            }
            const mc = document.getElementById('metric-method-count');
            if (mc) mc.innerText = methods.length || 0;
        } catch (_) {}
    }

    /* ------------------------------------------------------------------
       Export Helpers
       ------------------------------------------------------------------ */
    exportLogs() {
        const c = document.getElementById('log-container');
        if (!c) return;
        const text = Array.from(c.children).map(el => el.textContent).join('\n');
        this._download('puffin_logs.txt', text);
    }

    exportMetrics() {
        if (!this.chart) return;
        const labels = this.chart.data.labels;
        const ds = this.chart.data.datasets;
        let csv = 'Generation,CompressionRatePct,BenchmarkKB,ComplexityValue\n';
        labels.forEach((g, i) => {
            csv += `${g},${ds[0].data[i] || 0},${ds[1].data[i] || 0},${ds[2].data[i] || 0}\n`;
        });
        this._download('puffin_metrics.csv', csv);
    }

    _download(filename, content) {
        const blob = new Blob([content], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    }
}

/* Global helpers used by validation tab inline onclick */
async function loadValidationCheckpoints() {
    const sel = document.getElementById('validation-checkpoint-select');
    if (!sel) return;
    try {
        const res = await fetch('/api/checkpoint/list');
        const data = await res.json();
        if (data.checkpoints && data.checkpoints.length > 0) {
            sel.innerHTML = data.checkpoints.map(cp =>
                `<option value="${cp.name || cp.checkpoint_name}">${cp.name || cp.checkpoint_name} (Gen ${cp.generation || '?'})</option>`
            ).join('');
        } else {
            sel.innerHTML = '<option>No checkpoints available</option>';
        }
    } catch (_) { sel.innerHTML = '<option>Error loading</option>'; }
}

async function testCheckpoint() {
    const sel = document.getElementById('validation-checkpoint-select');
    const fileInput = document.getElementById('test-file-upload');
    const results = document.getElementById('validation-results');
    if (!sel || !fileInput || !results) return;
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a test file first.');
        return;
    }
    results.innerHTML = '<div class="placeholder-center"><div style="font-size:2rem;">⏳</div>Running test...</div>';
    const formData = new FormData();
    formData.append('checkpoint', sel.value);
    formData.append('file', fileInput.files[0]);
    try {
        const res = await fetch('/api/checkpoint/test', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) {
            results.innerHTML = `<div class="placeholder-center" style="color:var(--danger-color);">${data.error}</div>`;
        } else {
            const integrityColor = data.integrity === 'PASS' ? 'var(--success-color)' : 'var(--danger-color)';
            const savingsColor = data.savings_pct && data.savings_pct.startsWith('+') ? 'var(--danger-color)' : 'var(--success-color)';
            results.innerHTML = `
                <div style="padding:16px;">
                    <div style="font-size:15px;font-weight:600;margin-bottom:12px;color:var(--success-color);">✅ Test Complete</div>
                    <div class="info-row"><span>Original Size:</span><span>${data.original_size || '—'}</span></div>
                    <div class="info-row"><span>Compressed Size:</span><span>${data.compressed_size || '—'}</span></div>
                    <div class="info-row"><span>Ratio:</span><span>${data.ratio || '—'}</span></div>
                    <div class="info-row"><span>Savings:</span><span style="color:${savingsColor};font-weight:600;">${data.savings_pct || '—'}</span></div>
                    <div class="info-row"><span>Method:</span><span>${data.method || '—'}</span></div>
                    <div class="info-row"><span>Time:</span><span>${data.time_ms || '—'}</span></div>
                    <div class="info-row"><span>Round-trip Integrity:</span><span style="color:${integrityColor};font-weight:600;">${data.integrity || '—'}</span></div>
                    <div class="info-row"><span>Agent Fitness:</span><span>${data.agent_fitness || '—'}</span></div>
                    <div class="info-row"><span>Checkpoint Gen:</span><span>${data.checkpoint_generation || '—'}</span></div>
                </div>`;
        }
    } catch (e) {
        results.innerHTML = `<div class="placeholder-center" style="color:var(--danger-color);">Error: ${e.message}</div>`;
    }
}

const app = new PuffinZipAIApp();
window._pzApp = app;  // expose for onclick handlers
