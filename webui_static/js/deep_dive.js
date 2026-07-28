/* ============================================================================
   PuffinZipAI Web UI — Evolution Deep Dive Tab
   ============================================================================
   Provides gene pool visualization, breeding relationship graphs, mutation
   tracking, per-generation agent fitness analysis, interactive lineage
   highlighting, breeding network SVG flow, pool-to-pool breeding matrix,
   top breeders list, and method direction trend chart.
   ============================================================================ */

class DeepDiveManager {
    constructor() {
        this.ddChart = null;
        this.methodChart = null;
        this.ddPage = 1;
        this.ddTotalPages = 1;
        this.ddViewGenIndex = -1; // -1 = latest
        this.ddGenerations = [];  // cached from last poll
        this.expandedDDGens = new Set();
        this.pollInterval = null;
        this.selectedAgentId = null; // for lineage highlighting

        // Cross-generation lineage lookup maps (rebuilt on each poll)
        this._allAgentsById = {};      // agent_id → {fitness, generation, pool_index, parent_ids, ...}
        this._childrenOfAgent = {};    // agent_id → [child agent_ids] across ALL gens
        this._agentGeneration = {};    // agent_id → generation number where the agent appears

        this.init();
    }

    /* --- Gene pool colour palette (12 distinct hues) --- */
    static POOL_COLORS = [
        '#6366f1', // indigo
        '#f43f5e', // rose
        '#22d3ee', // cyan
        '#f59e0b', // amber
        '#10b981', // emerald
        '#a855f7', // purple
        '#3b82f6', // blue
        '#ef4444', // red
        '#84cc16', // lime
        '#ec4899', // pink
        '#14b8a6', // teal
        '#f97316', // orange
    ];

    static poolColor(index) {
        return DeepDiveManager.POOL_COLORS[index % DeepDiveManager.POOL_COLORS.length];
    }

    init() {
        this.initDDChart();
        this.initMethodChart();
        this.setupEventListeners();
        // Poll every 4s when the deep-dive tab is visible
        this.pollInterval = setInterval(() => {
            const tab = document.getElementById('deep-dive-top-tab');
            if (tab && tab.classList.contains('active')) {
                this.pollDeepDive();
            }
        }, 4000);
    }

    /* ------------------------------------------------------------------
       Deep Dive Chart — Best/Avg Fitness + Crossovers + Mutations
       ------------------------------------------------------------------ */
    initDDChart() {
        const ctx = document.getElementById('ddFitnessChart');
        if (!ctx) return;
        const existing = Chart.getChart(ctx);
        if (existing) existing.destroy();

        this.ddChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Best Fitness',
                        data: [],
                        borderColor: '#4ade80',
                        backgroundColor: 'rgba(74,222,128,0.1)',
                        tension: 0.3, borderWidth: 2.5, pointRadius: 3,
                        pointBackgroundColor: '#4ade80',
                        fill: true, yAxisID: 'y', type: 'line',
                        order: 1
                    },
                    {
                        label: 'Avg Fitness',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139,92,246,0.05)',
                        tension: 0.3, borderWidth: 2, pointRadius: 0,
                        fill: false, borderDash: [4, 4], yAxisID: 'y', type: 'line',
                        order: 2
                    },
                    {
                        label: 'Crossovers',
                        data: [],
                        borderColor: 'rgba(34,211,238,0.6)',
                        backgroundColor: 'rgba(34,211,238,0.25)',
                        borderWidth: 1,
                        yAxisID: 'y2',
                        barPercentage: 0.5,
                        categoryPercentage: 0.8,
                        order: 3
                    },
                    {
                        label: 'Mutations',
                        data: [],
                        borderColor: 'rgba(245,158,11,0.6)',
                        backgroundColor: 'rgba(245,158,11,0.3)',
                        borderWidth: 1,
                        yAxisID: 'y2',
                        barPercentage: 0.5,
                        categoryPercentage: 0.8,
                        order: 4
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#f8f8f2', boxWidth: 12, font: { size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Generation ${items[0]?.label || '?'}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#333' },
                        ticks: { color: '#f8f8f2', maxTicksLimit: 25 }
                    },
                    y: {
                        grid: { color: '#333' }, position: 'left',
                        ticks: { color: '#4ade80' },
                        title: { display: true, text: 'Fitness', color: '#4ade80' }
                    },
                    y2: {
                        grid: { drawOnChartArea: false }, beginAtZero: true, position: 'right',
                        ticks: { color: '#22d3ee', stepSize: 1 },
                        title: { display: true, text: 'Count', color: '#22d3ee' }
                    }
                }
            }
        });
    }

    /* ------------------------------------------------------------------
       Method Direction Chart — RLE min run & novel method adoption
       ------------------------------------------------------------------ */
    initMethodChart() {
        const ctx = document.getElementById('ddMethodChart');
        if (!ctx) return;
        const existing = Chart.getChart(ctx);
        if (existing) existing.destroy();

        this.methodChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Avg RLE Min Run',
                        data: [],
                        borderColor: '#22d3ee',
                        backgroundColor: 'rgba(34,211,238,0.1)',
                        tension: 0.3, borderWidth: 2.5, pointRadius: 3,
                        pointBackgroundColor: '#22d3ee',
                        fill: true, yAxisID: 'y', type: 'line',
                        order: 1
                    },
                    {
                        label: 'Avg Recipe Strength',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.1)',
                        tension: 0.3, borderWidth: 2, pointRadius: 2,
                        pointBackgroundColor: '#f59e0b',
                        borderDash: [4, 3],
                        fill: false, yAxisID: 'y', type: 'line',
                        order: 0
                    },
                    {
                        label: 'Cross-Pool Breeds',
                        data: [],
                        borderColor: 'rgba(168,85,247,0.7)',
                        backgroundColor: 'rgba(168,85,247,0.3)',
                        borderWidth: 1,
                        yAxisID: 'y2',
                        barPercentage: 0.4,
                        categoryPercentage: 0.8,
                        order: 3
                    },
                    {
                        label: 'Novel Methods',
                        data: [],
                        borderColor: 'rgba(244,63,94,0.7)',
                        backgroundColor: 'rgba(244,63,94,0.3)',
                        borderWidth: 1,
                        yAxisID: 'y2',
                        barPercentage: 0.4,
                        categoryPercentage: 0.8,
                        order: 4
                    },
                    {
                        label: 'Unique Families',
                        data: [],
                        borderColor: 'rgba(16,185,129,0.7)',
                        backgroundColor: 'rgba(16,185,129,0.3)',
                        borderWidth: 1,
                        yAxisID: 'y2',
                        barPercentage: 0.4,
                        categoryPercentage: 0.8,
                        order: 5
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#f8f8f2', boxWidth: 12, font: { size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Generation ${items[0]?.label || '?'}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#333' },
                        ticks: { color: '#f8f8f2', maxTicksLimit: 25 }
                    },
                    y: {
                        grid: { color: '#333' }, position: 'left',
                        ticks: { color: '#22d3ee' },
                        title: { display: true, text: 'RLE / Strength', color: '#22d3ee' },
                        beginAtZero: true, max: 10
                    },
                    y2: {
                        grid: { drawOnChartArea: false }, beginAtZero: true, position: 'right',
                        ticks: { color: '#a855f7', stepSize: 1 },
                        title: { display: true, text: 'Count', color: '#a855f7' }
                    }
                }
            }
        });
    }

    /* ------------------------------------------------------------------
       Event Listeners
       ------------------------------------------------------------------ */
    setupEventListeners() {
        const q = id => document.getElementById(id);

        q('dd-page-prev')?.addEventListener('click', () => {
            if (this.ddPage > 1) { this.ddPage--; this.pollDeepDive(); }
        });
        q('dd-page-next')?.addEventListener('click', () => {
            if (this.ddPage < this.ddTotalPages) { this.ddPage++; this.pollDeepDive(); }
        });
        q('dd-gen-prev')?.addEventListener('click', () => {
            if (this.ddViewGenIndex > 0) {
                this.ddViewGenIndex--;
                this.renderGenePoolGrid();
            } else if (this.ddViewGenIndex === -1 && this.ddGenerations.length > 1) {
                this.ddViewGenIndex = this.ddGenerations.length - 2;
                this.renderGenePoolGrid();
            }
        });
        q('dd-gen-next')?.addEventListener('click', () => {
            if (this.ddViewGenIndex >= 0 && this.ddViewGenIndex < this.ddGenerations.length - 1) {
                this.ddViewGenIndex++;
                this.renderGenePoolGrid();
            } else {
                this.ddViewGenIndex = -1;
                this.renderGenePoolGrid();
            }
        });
        q('dd-show-fitness')?.addEventListener('change', () => {
            this.renderGenePoolGrid();
        });
        q('dd-chart-refresh')?.addEventListener('click', () => {
            this.pollDeepDive();
        });
        q('dd-lineage-clear')?.addEventListener('click', () => {
            this.clearLineageSelection();
        });
    }

    /* ------------------------------------------------------------------
       Main Poll — fetch /api/evolution/deep-dive
       ------------------------------------------------------------------ */
    async pollDeepDive() {
        try {
            const res = await fetch(`${_P}/api/evolution/deep-dive?page=${this.ddPage}&per_page=20`);
            const d = await res.json();

            this.ddTotalPages = d.total_pages || 1;
            this.ddGenerations = d.generations || [];

            // Rebuild cross-generation lineage maps
            this._rebuildLineageMaps();

            const q = id => document.getElementById(id);
            const set = (id, v) => { const el = q(id); if (el) el.innerText = v; };

            // KPIs
            set('dd-pool-count', d.total_pool_count || 0);
            set('dd-stagnation', d.stagnation_counter || 0);
            const mutRate = d.mutation_rate || 0;
            set('dd-mutation-rate', (mutRate * 100).toFixed(0) + '%');

            // Best overall from top agents
            if (d.top_agents && d.top_agents.length > 0) {
                const best = d.top_agents[0].fitness || 0;
                set('dd-best-fitness', parseFloat(best).toFixed(4));
            }

            // Page info
            const pi = q('dd-page-info');
            if (pi) pi.textContent = `Page ${d.page || 1} / ${this.ddTotalPages}`;

            // Update chart
            this.updateDDChart(d.generations);

            // Gene pool legend
            this.renderGenePools(d.gene_pools || {});

            // Top agents
            this.renderTopAgents(d.top_agents || []);

            // Gene pool grid (latest generation by default)
            this.renderGenePoolGrid();

            // Re-render lineage detail if an agent is selected (data may have updated)
            if (this.selectedAgentId) this.renderLineageDetail();

            // Breeding network visualizations
            this.renderBreedingFlow();
            this.renderPoolMatrix();
            this.renderTopBreeders(d.top_breeders || []);

            // Method direction
            this.updateMethodChart(d.method_direction || []);
            // v0.9.10: Store method registry summary for use in renderMethodSummary
            this._methodRegistry = d.method_registry || {};
            this.renderMethodSummary(d.method_direction || []);

            // Generations detail
            this.renderGenerationsDetail(d.generations);

        } catch (e) {
            console.warn('Deep dive poll error:', e);
        }
    }

    /* ------------------------------------------------------------------
       Cross-Generation Lineage Map Builder
       ------------------------------------------------------------------ */
    _rebuildLineageMaps() {
        this._allAgentsById = {};
        this._childrenOfAgent = {};
        this._agentGeneration = {};

        for (const gen of this.ddGenerations) {
            const genNum = gen.generation || 0;
            for (const agent of (gen.agents || [])) {
                const aid = agent.agent_id;
                if (!aid) continue;
                this._allAgentsById[aid] = { ...agent, _gen: genNum };
                this._agentGeneration[aid] = genNum;

                // Build children-of map from parent_ids
                const pids = agent.parent_ids || [];
                for (const pid of pids) {
                    if (!this._childrenOfAgent[pid]) this._childrenOfAgent[pid] = [];
                    this._childrenOfAgent[pid].push(aid);
                }
            }
        }
    }

    /* ------------------------------------------------------------------
       Chart Update
       ------------------------------------------------------------------ */
    updateDDChart(generations) {
        if (!this.ddChart || !generations || generations.length === 0) return;

        this.ddChart.data.labels = generations.map(g => g.generation);
        this.ddChart.data.datasets[0].data = generations.map(g => g.best_fitness ?? 0);
        this.ddChart.data.datasets[1].data = generations.map(g => g.avg_fitness ?? 0);
        this.ddChart.data.datasets[2].data = generations.map(g => g.crossover_count ?? 0);
        this.ddChart.data.datasets[3].data = generations.map(g => g.mutation_count ?? 0);
        this.ddChart.update('none');
    }

    /* ------------------------------------------------------------------
       Gene Pool Legend Render
       ------------------------------------------------------------------ */
    renderGenePools(pools) {
        const container = document.getElementById('dd-gene-pool-legend');
        if (!container) return;

        const entries = Object.values(pools);
        if (entries.length === 0) {
            container.innerHTML = '<div class="placeholder-cell">No gene pools yet</div>';
            return;
        }

        container.innerHTML = entries.map(p => {
            const color = DeepDiveManager.poolColor(p.index);
            const rootShort = (p.root_ancestor || '').substring(0, 12);
            // Specialization badge
            const specIcons = { compression: '📦', robustness: '🛡️', hybrid: '🔀' };
            const specIcon = specIcons[p.specialization] || '📦';
            const specLabel = p.specialization || 'compression';
            // Dual fitness scores
            const avgComp = (p.avg_compression || 0).toFixed(2);
            const avgRob = (p.avg_robustness || 0).toFixed(2);
            const compAgents = p.compression_agents || 0;
            const robAgents = p.robustness_agents || 0;
            // v0.9.10: Recipe method info
            const domFamily = p.dominant_family || 'none';
            const avgStr = (p.avg_recipe_strength || 0).toFixed(2);
            const matureCount = p.mature_recipe_count || 0;
            const familyCount = Object.keys(p.recipe_families || {}).length;
            const resurrectedCount = p.resurrected_agent_count || 0;
            // Build recipe family distribution mini-bar
            const families = p.recipe_families || {};
            const familyEntries = Object.entries(families).sort((a,b) => b[1] - a[1]).slice(0, 3);
            const familyBar = familyEntries.length > 0
                ? familyEntries.map(([f, c]) => `<span style="font-size:9px;opacity:.8">${f}:${c}</span>`).join(' ')
                : '<span style="font-size:9px;opacity:.5">no methods</span>';
            const resurrectBadge = resurrectedCount > 0
                ? `<span style="font-size:9px;color:#f59e0b" title="${resurrectedCount} resurrected agents">🧟${resurrectedCount}</span>`
                : '';
            return `<div class="dd-pool-item" title="${p.root_ancestor}\nCompression: ${avgComp} (${compAgents} agents)\nRobustness: ${avgRob} (${robAgents} agents)\nDominant Method: ${domFamily}\nAvg Strength: ${avgStr}\nMature Methods: ${matureCount}\nFamilies: ${familyCount}\nResurrected: ${resurrectedCount}">
                <div class="dd-pool-swatch" style="background:${color};"></div>
                <span class="dd-pool-label">${specIcon} Pool ${p.index + 1}</span>
                <span class="dd-pool-count">${p.size} <small style="opacity:.7">C:${avgComp} R:${avgRob}</small></span>
                <div style="font-size:9px;margin-top:2px;display:flex;gap:4px;flex-wrap:wrap;align-items:center;">
                    <span style="color:#f59e0b" title="Avg recipe strength">💪${avgStr}</span>
                    <span style="color:#f43f5e" title="${matureCount} mature novel methods">🧬${matureCount}</span>
                    ${resurrectBadge}
                    ${familyBar}
                </div>
            </div>`;
        }).join('');
    }

    /* ------------------------------------------------------------------
       Top Agents Render
       ------------------------------------------------------------------ */
    renderTopAgents(agents) {
        const container = document.getElementById('dd-top-agents');
        if (!container) return;

        if (agents.length === 0) {
            container.innerHTML = '<div class="placeholder-cell">No agents yet</div>';
            return;
        }

        container.innerHTML = agents.slice(0, 15).map((a, i) => {
            const color = DeepDiveManager.poolColor(a.pool_index || 0);
            const fit = (a.fitness !== null && a.fitness !== undefined && a.fitness > -999)
                ? parseFloat(a.fitness).toFixed(4) : '—';
            const idShort = (a.agent_id || '').substring(0, 16);
            const rankColors = ['#ffd700', '#c0c0c0', '#cd7f32']; // gold, silver, bronze
            const rankBg = i < 3 ? rankColors[i] : 'var(--bg-surface-light)';
            return `<div class="dd-top-agent-item">
                <div class="dd-top-rank" style="background:${rankBg};">${i + 1}</div>
                <div class="dd-top-agent-pool-swatch" style="background:${color};" title="Pool ${(a.pool_index || 0) + 1}"></div>
                <span class="dd-top-agent-id" title="${a.agent_id}">${idShort}</span>
                <span class="dd-top-agent-fitness">${fit}</span>
            </div>`;
        }).join('');
    }

    /* ------------------------------------------------------------------
       Gene Pool Grid — Coloured Agent Squares with Click-to-Highlight Lineage
       ------------------------------------------------------------------ */
    renderGenePoolGrid() {
        const container = document.getElementById('dd-gene-pool-grid');
        const genLabel = document.getElementById('dd-gen-label');
        if (!container) return;

        if (this.ddGenerations.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:40px;text-align:center;">Start training to see agents...</div>';
            if (genLabel) genLabel.textContent = 'No data';
            return;
        }

        const genIdx = this.ddViewGenIndex === -1
            ? this.ddGenerations.length - 1
            : Math.min(this.ddViewGenIndex, this.ddGenerations.length - 1);
        const gen = this.ddGenerations[genIdx];
        if (genLabel) genLabel.textContent = `Gen ${gen.generation}`;

        const agents = gen.agents || [];
        const showFitness = document.getElementById('dd-show-fitness')?.checked;
        const breedingPairs = gen.breeding_pairs || [];

        if (agents.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:40px;text-align:center;">No agents in this generation</div>';
            return;
        }

        // Build reverse maps: agent → parents, agent → children (within this gen)
        const childrenOf = {};  // parentId → [childIds]
        const parentsOf = {};   // childId → [parentIds]
        for (const bp of breedingPairs) {
            const p1 = bp.parent1, p2 = bp.parent2, ch = bp.child;
            if (!childrenOf[p1]) childrenOf[p1] = [];
            if (!childrenOf[p2]) childrenOf[p2] = [];
            childrenOf[p1].push(ch);
            childrenOf[p2].push(ch);
            parentsOf[ch] = [p1, p2];
        }

        // Sort by fitness descending for ranking
        const sorted = [...agents].sort((a, b) => (b.fitness || -9999) - (a.fitness || -9999));

        // Build set of agent IDs in the current grid for quick lookup
        const gridAgentIds = new Set(sorted.map(a => a.agent_id));

        // Determine lineage highlights using CROSS-GENERATION lookups
        const selected = this.selectedAgentId;
        const highlightParents = new Set();
        const highlightChildren = new Set();
        const highlightSiblings = new Set();
        if (selected) {
            // --- Parents of the selected agent (from cross-gen lookup) ---
            const selData = this._allAgentsById[selected];
            const selParentIds = selData ? (selData.parent_ids || []) : [];
            selParentIds.forEach(p => highlightParents.add(p));

            // Also add parents from same-gen breeding_pairs
            (parentsOf[selected] || []).forEach(p => highlightParents.add(p));

            // --- Children of the selected agent (from cross-gen lookup) ---
            const crossGenChildren = this._childrenOfAgent[selected] || [];
            crossGenChildren.forEach(c => highlightChildren.add(c));

            // Also add children from same-gen breeding_pairs
            (childrenOf[selected] || []).forEach(c => highlightChildren.add(c));

            // Check all agents in the current grid: if selected appears in their parent_ids
            for (const a of sorted) {
                if ((a.parent_ids || []).includes(selected)) {
                    highlightChildren.add(a.agent_id);
                }
            }

            // --- Siblings: agents sharing at least one parent with selected ---
            if (selParentIds.length > 0) {
                for (const a of sorted) {
                    if (a.agent_id === selected) continue;
                    const aPids = a.parent_ids || [];
                    if (aPids.some(p => selParentIds.includes(p))) {
                        highlightSiblings.add(a.agent_id);
                    }
                }
            }
        }

        container.innerHTML = sorted.map((a, rank) => {
            const color = DeepDiveManager.poolColor(a.pool_index || 0);
            const fit = (a.fitness !== null && a.fitness !== undefined && a.fitness > -999)
                ? parseFloat(a.fitness).toFixed(4) : '—';
            const isElite = a.is_elite;
            let classes = 'dd-agent-cell';
            if (isElite) classes += ' elite-agent';

            // Lineage highlighting classes
            if (selected === a.agent_id) classes += ' lineage-selected';
            else if (highlightParents.has(a.agent_id)) classes += ' lineage-parent';
            else if (highlightChildren.has(a.agent_id)) classes += ' lineage-child';
            else if (highlightSiblings.has(a.agent_id)) classes += ' lineage-sibling';
            else if (selected) classes += ' lineage-dimmed';

            // Breeding indicator: was this agent bred (has 2+ parents)?
            const wasBred = (a.parent_ids || []).length >= 2;
            if (wasBred && !selected) classes += ' was-bred';

            const label = showFitness ? fit : `#${rank + 1}`;
            const idShort = (a.agent_id || '').substring(0, 14);
            const parentStr = (a.parent_ids || []).map(p => p.substring(0, 12)).join(' + ') || 'None (seed)';
            const tooltipData = encodeURIComponent(JSON.stringify({
                id: idShort, fit, rank: rank + 1, total: sorted.length,
                pool: (a.pool_index || 0) + 1, color, genBorn: a.generation_born || 0,
                parents: parentStr, lr: (a.learning_rate || 0).toFixed(4),
                explore: (a.exploration_rate || 0).toFixed(4), elite: isElite,
                agentId: a.agent_id, wasBred,
                childCount: (this._childrenOfAgent[a.agent_id] || []).length,
                // v0.9.10: recipe info
                recipeFamily: a.recipe_family || 'none',
                recipeStrength: (a.recipe_strength || 0).toFixed(2),
                recipeImprovements: a.recipe_improvements || 0,
                hasNovelMethod: a.has_novel_method || false,
                novelPipeline: a.novel_pipeline || 'none',
                recipeRediscovered: a.recipe_times_rediscovered || 0,
            }));

            return `<div class="${classes}" style="background:${color};"
                         data-agent-id="${a.agent_id}"
                         data-tooltip="${tooltipData}"
                         onmouseenter="window._ddMgr.showTooltip(event, this)"
                         onmouseleave="window._ddMgr.hideTooltip()"
                         onclick="window._ddMgr.selectAgent('${a.agent_id}')">
                <span class="dd-agent-rank">${label}</span>
            </div>`;
        }).join('');

        // Update breeding flow SVG for current gen too
        this.renderBreedingFlow();
    }

    /* ------------------------------------------------------------------
       Agent Click → Lineage Selection
       ------------------------------------------------------------------ */
    selectAgent(agentId) {
        if (this.selectedAgentId === agentId) {
            this.clearLineageSelection();
            return;
        }
        this.selectedAgentId = agentId;
        const infoBar = document.getElementById('dd-lineage-info');
        const agentLabel = document.getElementById('dd-lineage-agent-id');
        if (infoBar) infoBar.style.display = 'block';
        if (agentLabel) agentLabel.textContent = agentId.substring(0, 20);
        this.renderGenePoolGrid();
        this.renderLineageDetail();
    }

    clearLineageSelection() {
        this.selectedAgentId = null;
        const infoBar = document.getElementById('dd-lineage-info');
        if (infoBar) infoBar.style.display = 'none';
        const detailPanel = document.getElementById('dd-lineage-detail');
        if (detailPanel) detailPanel.style.display = 'none';
        this.renderGenePoolGrid();
    }

    /* ------------------------------------------------------------------
       Navigate to the generation containing a specific agent
       ------------------------------------------------------------------ */
    navigateToAgent(agentId) {
        const genNum = this._agentGeneration[agentId];
        if (genNum === undefined) return;
        // Find the index of this generation in ddGenerations
        const idx = this.ddGenerations.findIndex(g => g.generation === genNum);
        if (idx === -1) return;
        this.ddViewGenIndex = idx;
        this.selectedAgentId = agentId;
        const infoBar = document.getElementById('dd-lineage-info');
        const agentLabel = document.getElementById('dd-lineage-agent-id');
        if (infoBar) infoBar.style.display = 'block';
        if (agentLabel) agentLabel.textContent = agentId.substring(0, 20);
        this.renderGenePoolGrid();
        this.renderLineageDetail();
    }

    /* ------------------------------------------------------------------
       Lineage Detail Panel — Multi-generational ancestry & descendants
       ------------------------------------------------------------------ */
    renderLineageDetail() {
        const panel = document.getElementById('dd-lineage-detail');
        if (!panel) return;

        const selected = this.selectedAgentId;
        if (!selected) {
            panel.style.display = 'none';
            return;
        }

        panel.style.display = 'block';
        const selData = this._allAgentsById[selected];
        if (!selData) {
            panel.innerHTML = `<div class="dd-lineage-detail-inner">
                <div class="dd-lineage-detail-title">Agent not found in cached generations</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">
                    The agent <span style="font-family:monospace;color:var(--text-primary);">${selected.substring(0, 20)}</span>
                    may be from a generation outside the current page. Try navigating to a different page.
                </div>
            </div>`;
            return;
        }

        const selFit = (selData.fitness !== null && selData.fitness !== undefined && selData.fitness > -999)
            ? parseFloat(selData.fitness).toFixed(4) : '—';
        const selGen = selData._gen || selData.generation_born || 0;
        const selPoolIdx = selData.pool_index || 0;
        const selColor = DeepDiveManager.poolColor(selPoolIdx);

        // --- Build ancestry chain (parents, grandparents, great-grandparents) ---
        const ancestors = [];  // [{depth: 1, agents: [...]}]
        let currentParentIds = selData.parent_ids || [];
        const depthLabels = ['Parents', 'Grandparents', 'Great-Grandparents', 'Great²-Grandparents'];
        for (let depth = 0; depth < 4 && currentParentIds.length > 0; depth++) {
            const depthAgents = [];
            const nextParentIds = [];
            for (const pid of currentParentIds) {
                const pData = this._allAgentsById[pid];
                if (pData) {
                    depthAgents.push(pData);
                    (pData.parent_ids || []).forEach(gp => nextParentIds.push(gp));
                } else {
                    depthAgents.push({ agent_id: pid, fitness: null, _gen: null, pool_index: 0, _unknown: true });
                }
            }
            ancestors.push({ label: depthLabels[depth] || `Ancestor Depth ${depth + 1}`, agents: depthAgents });
            currentParentIds = [...new Set(nextParentIds)];
        }

        // --- Children (direct offspring) ---
        const childIds = this._childrenOfAgent[selected] || [];
        const children = childIds.map(cid => {
            const cData = this._allAgentsById[cid];
            return cData || { agent_id: cid, fitness: null, _gen: null, pool_index: 0, _unknown: true };
        });

        // --- Siblings (agents sharing at least one parent, across all gens) ---
        const selParentIds = selData.parent_ids || [];
        const siblingIds = new Set();
        for (const pid of selParentIds) {
            const pChildren = this._childrenOfAgent[pid] || [];
            pChildren.forEach(cid => {
                if (cid !== selected) siblingIds.add(cid);
            });
        }
        const siblings = [...siblingIds].map(sid => {
            const sData = this._allAgentsById[sid];
            return sData || { agent_id: sid, fitness: null, _gen: null, pool_index: 0, _unknown: true };
        }).slice(0, 10); // cap for readability

        // --- Render HTML ---
        const renderAgentChip = (a, role) => {
            const aid = a.agent_id || '';
            const idShort = aid.substring(0, 16);
            const fit = (a.fitness !== null && a.fitness !== undefined && a.fitness > -999)
                ? parseFloat(a.fitness).toFixed(4) : '?';
            const gen = a._gen != null ? `Gen ${a._gen}` : '?';
            const color = DeepDiveManager.poolColor(a.pool_index || 0);
            const unknown = a._unknown ? ' opacity:0.5;' : '';
            const clickable = !a._unknown ? `onclick="window._ddMgr.navigateToAgent('${aid}')" style="cursor:pointer;${unknown}"` : `style="${unknown}"`;
            const roleColor = role === 'parent' ? '#f59e0b' : role === 'child' ? '#22d3ee' : '#94a3b8';
            return `<div class="dd-lineage-chip" ${clickable} title="${aid}\nFitness: ${fit}\n${gen}">
                <span class="dd-lineage-chip-dot" style="background:${color};"></span>
                <span class="dd-lineage-chip-id">${idShort}</span>
                <span class="dd-lineage-chip-fit" style="color:${roleColor};">${fit}</span>
                <span class="dd-lineage-chip-gen">${gen}</span>
            </div>`;
        };

        let html = `<div class="dd-lineage-detail-inner">`;

        // Selected agent summary
        html += `<div class="dd-lineage-selected-summary">
            <span class="dd-lineage-chip-dot" style="background:${selColor};width:12px;height:12px;"></span>
            <span style="font-weight:700;font-family:monospace;color:var(--text-primary);">${selected.substring(0, 20)}</span>
            <span style="color:#4ade80;font-weight:600;margin-left:8px;">Fitness: ${selFit}</span>
            <span style="color:var(--text-muted);margin-left:8px;">Gen ${selGen}</span>
            <span style="color:var(--text-muted);margin-left:8px;">Pool ${selPoolIdx + 1}</span>
        </div>`;

        // Ancestors
        if (ancestors.length > 0) {
            for (const level of ancestors) {
                html += `<div class="dd-lineage-section">
                    <div class="dd-lineage-section-label" style="color:#f59e0b;">⬆ ${level.label}</div>
                    <div class="dd-lineage-chip-row">${level.agents.map(a => renderAgentChip(a, 'parent')).join('')}</div>
                </div>`;
            }
        } else {
            html += `<div class="dd-lineage-section">
                <div class="dd-lineage-section-label" style="color:#f59e0b;">⬆ Parents</div>
                <div class="dd-lineage-chip-row"><span style="color:var(--text-muted);font-size:11px;">None (seed agent)</span></div>
            </div>`;
        }

        // Children
        html += `<div class="dd-lineage-section">
            <div class="dd-lineage-section-label" style="color:#22d3ee;">⬇ Children (${children.length})</div>
            <div class="dd-lineage-chip-row">${children.length > 0
                ? children.slice(0, 12).map(a => renderAgentChip(a, 'child')).join('')
                    + (children.length > 12 ? `<span class="dd-lineage-more">+${children.length - 12} more</span>` : '')
                : '<span style="color:var(--text-muted);font-size:11px;">No offspring recorded</span>'
            }</div>
        </div>`;

        // Siblings
        if (siblings.length > 0) {
            html += `<div class="dd-lineage-section">
                <div class="dd-lineage-section-label" style="color:#94a3b8;">↔ Siblings (${siblingIds.size})</div>
                <div class="dd-lineage-chip-row">${siblings.map(a => renderAgentChip(a, 'sibling')).join('')}
                    ${siblingIds.size > 10 ? `<span class="dd-lineage-more">+${siblingIds.size - 10} more</span>` : ''}
                </div>
            </div>`;
        }

        html += `</div>`;
        panel.innerHTML = html;
    }

    /* ------------------------------------------------------------------
       Fixed-Position Tooltip (avoids overflow clipping)
       ------------------------------------------------------------------ */
    showTooltip(event, cell) {
        let tip = document.getElementById('dd-floating-tooltip');
        if (!tip) {
            tip = document.createElement('div');
            tip.id = 'dd-floating-tooltip';
            tip.className = 'dd-agent-tooltip';
            document.body.appendChild(tip);
        }
        try {
            const d = JSON.parse(decodeURIComponent(cell.dataset.tooltip));
            const breedInfo = d.wasBred
                ? `<div class="dd-tooltip-row"><span class="dd-tooltip-label">Origin</span><span class="dd-tooltip-value" style="color:#f59e0b;">🧬 Bred</span></div>`
                : `<div class="dd-tooltip-row"><span class="dd-tooltip-label">Origin</span><span class="dd-tooltip-value" style="color:#94a3b8;">Seed / Clone</span></div>`;
            const childInfo = d.childCount > 0
                ? `<div class="dd-tooltip-row"><span class="dd-tooltip-label">Offspring</span><span class="dd-tooltip-value" style="color:#22d3ee;">${d.childCount} children</span></div>`
                : '';

            // v0.9.10: Recipe/method info
            const recipeFamily = d.recipeFamily || 'none';
            const recipeStrength = d.recipeStrength || '0.00';
            const strengthPct = Math.round(parseFloat(recipeStrength) * 100);
            const strengthColor = strengthPct >= 70 ? '#4ade80' : strengthPct >= 30 ? '#f59e0b' : '#f43f5e';
            const novelBadge = d.hasNovelMethod
                ? '<span style="color:#f43f5e;font-weight:700;">✨ NOVEL</span>'
                : `<span style="color:#94a3b8;">${d.recipeImprovements || 0} improvements</span>`;
            const resurrectBadge = d.recipeRediscovered > 0
                ? `<div class="dd-tooltip-row"><span class="dd-tooltip-label">Resurrected</span><span class="dd-tooltip-value" style="color:#f59e0b;">🧟 ${d.recipeRediscovered}x</span></div>`
                : '';

            tip.innerHTML = `
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Agent</span><span class="dd-tooltip-value">${d.id}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Fitness</span><span class="dd-tooltip-value" style="color:#4ade80;">${d.fit}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Rank</span><span class="dd-tooltip-value">#${d.rank} of ${d.total}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Pool</span><span class="dd-tooltip-value"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${d.color};vertical-align:middle;margin-right:4px;"></span>Pool ${d.pool}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Gen Born</span><span class="dd-tooltip-value">${d.genBorn}</span></div>
                ${breedInfo}
                ${childInfo}
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Parents</span><span class="dd-tooltip-value">${d.parents}</span></div>
                <div style="border-top:1px solid #444;margin:4px 0;"></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Method</span><span class="dd-tooltip-value" style="font-size:10px;">${recipeFamily}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Strength</span><span class="dd-tooltip-value" style="color:${strengthColor};">
                    <span style="display:inline-block;width:40px;height:6px;background:#333;border-radius:3px;vertical-align:middle;margin-right:4px;position:relative;">
                        <span style="display:block;width:${strengthPct}%;height:100%;background:${strengthColor};border-radius:3px;"></span>
                    </span>${recipeStrength}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Status</span><span class="dd-tooltip-value">${novelBadge}</span></div>
                ${resurrectBadge}
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">LR</span><span class="dd-tooltip-value">${d.lr}</span></div>
                <div class="dd-tooltip-row"><span class="dd-tooltip-label">Explore</span><span class="dd-tooltip-value">${d.explore}</span></div>
                ${d.elite ? '<div style="text-align:center;margin-top:4px;color:#ffd700;font-weight:700;">⭐ ELITE</div>' : ''}
                <div style="text-align:center;margin-top:6px;font-size:9px;color:var(--text-muted);">Click to trace lineage</div>
            `;
        } catch (_) { return; }

        tip.style.display = 'block';
        const rect = cell.getBoundingClientRect();
        const tipW = tip.offsetWidth || 220;
        const tipH = tip.offsetHeight || 200;
        let left = rect.left + rect.width / 2 - tipW / 2;
        let top = rect.top - tipH - 10;
        if (top < 4) top = rect.bottom + 10;
        left = Math.max(4, Math.min(left, window.innerWidth - tipW - 4));
        tip.style.left = left + 'px';
        tip.style.top = top + 'px';
    }

    hideTooltip() {
        const tip = document.getElementById('dd-floating-tooltip');
        if (tip) tip.style.display = 'none';
    }

    /* ------------------------------------------------------------------
       Breeding Flow SVG — Visual parent→child connections
       ------------------------------------------------------------------ */
    renderBreedingFlow() {
        const svg = document.getElementById('dd-breed-flow-svg');
        const placeholder = document.getElementById('dd-breed-flow-placeholder');
        const genLabel = document.getElementById('dd-breed-flow-gen-label');
        if (!svg) return;

        if (this.ddGenerations.length === 0) {
            svg.style.display = 'none';
            if (placeholder) placeholder.style.display = 'block';
            return;
        }

        const genIdx = this.ddViewGenIndex === -1
            ? this.ddGenerations.length - 1
            : Math.min(this.ddViewGenIndex, this.ddGenerations.length - 1);
        const gen = this.ddGenerations[genIdx];
        if (genLabel) genLabel.textContent = `Gen ${gen.generation}`;

        const pairs = gen.breeding_pairs || [];
        if (pairs.length === 0) {
            svg.style.display = 'none';
            if (placeholder) {
                placeholder.style.display = 'block';
                placeholder.textContent = 'No breeding in this generation';
            }
            return;
        }

        if (placeholder) placeholder.style.display = 'none';
        svg.style.display = 'block';

        // Collect unique agents involved in breeding
        const agentSet = new Set();
        for (const bp of pairs) {
            agentSet.add(bp.parent1);
            agentSet.add(bp.parent2);
            agentSet.add(bp.child);
        }
        const agentList = [...agentSet];

        // Layout: parents on left column, children on right column
        const parentIds = new Set();
        const childIds = new Set();
        for (const bp of pairs) {
            parentIds.add(bp.parent1);
            parentIds.add(bp.parent2);
            childIds.add(bp.child);
        }

        // Remove parents that are also children (edge case) from parent column
        const pureParents = [...parentIds].filter(id => !childIds.has(id) || parentIds.has(id));
        const pureChildren = [...childIds];

        // Limit to first 20 pairs max for readability
        const visiblePairs = pairs.slice(0, 20);
        // Recollect from visible pairs
        const vpParents = new Set();
        const vpChildren = new Set();
        for (const bp of visiblePairs) {
            vpParents.add(bp.parent1);
            vpParents.add(bp.parent2);
            vpChildren.add(bp.child);
        }
        const leftCol = [...vpParents];
        const rightCol = [...vpChildren];

        const rowH = 28;
        const padY = 16;
        const svgH = Math.max(leftCol.length, rightCol.length) * rowH + padY * 2;
        const svgW = svg.parentElement?.clientWidth || 500;
        svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
        svg.setAttribute('width', svgW);
        svg.setAttribute('height', svgH);

        const leftX = 120;
        const rightX = svgW - 120;
        const leftPositions = {};
        const rightPositions = {};

        let svgContent = '';

        // Draw parent nodes (left column)
        leftCol.forEach((id, i) => {
            const y = padY + i * rowH + rowH / 2;
            leftPositions[id] = { x: leftX, y };
            const poolIdx = this._getAgentPoolIndex(id);
            const color = DeepDiveManager.poolColor(poolIdx);
            const idShort = id.substring(0, 10);
            const fitStr = this._getAgentFitness(id);
            svgContent += `<circle cx="${leftX}" cy="${y}" r="8" fill="${color}" stroke="#fff" stroke-width="1.5" opacity="0.9"/>`;
            svgContent += `<text x="${leftX - 14}" y="${y + 4}" text-anchor="end" fill="#ccc" font-size="9" font-family="monospace">${idShort}</text>`;
            svgContent += `<text x="${leftX + 14}" y="${y + 3}" fill="#4ade80" font-size="8" font-family="monospace">${fitStr}</text>`;
        });

        // Draw child nodes (right column)
        rightCol.forEach((id, i) => {
            const y = padY + i * rowH + rowH / 2;
            rightPositions[id] = { x: rightX, y };
            const poolIdx = this._getAgentPoolIndex(id);
            const color = DeepDiveManager.poolColor(poolIdx);
            const idShort = id.substring(0, 10);
            const fitStr = this._getAgentFitness(id);
            svgContent += `<circle cx="${rightX}" cy="${y}" r="8" fill="${color}" stroke="#fff" stroke-width="1.5" opacity="0.9"/>`;
            svgContent += `<text x="${rightX + 14}" y="${y + 4}" fill="#ccc" font-size="9" font-family="monospace">${idShort}</text>`;
            svgContent += `<text x="${rightX - 14}" y="${y + 3}" text-anchor="end" fill="#4ade80" font-size="8" font-family="monospace">${fitStr}</text>`;
        });

        // Draw connection curves (parent → child)
        for (const bp of visiblePairs) {
            const p1Pos = leftPositions[bp.parent1];
            const p2Pos = leftPositions[bp.parent2];
            const childPos = rightPositions[bp.child];
            if (!p1Pos || !childPos) continue;

            const p1Color = DeepDiveManager.poolColor(bp.parent1_pool || 0);
            const p2Color = DeepDiveManager.poolColor(bp.parent2_pool || 0);
            const isCrossPool = bp.cross_pool;
            const strokeOpacity = isCrossPool ? 0.7 : 0.4;

            // Curve from parent1 → child
            const cx1 = (p1Pos.x + childPos.x) / 2;
            svgContent += `<path d="M${p1Pos.x + 8} ${p1Pos.y} C${cx1} ${p1Pos.y}, ${cx1} ${childPos.y}, ${childPos.x - 8} ${childPos.y}"
                fill="none" stroke="${p1Color}" stroke-width="${isCrossPool ? 2 : 1.2}" opacity="${strokeOpacity}"
                stroke-dasharray="${isCrossPool ? '4,3' : 'none'}"/>`;

            // Curve from parent2 → child
            if (p2Pos) {
                svgContent += `<path d="M${p2Pos.x + 8} ${p2Pos.y} C${cx1} ${p2Pos.y}, ${cx1} ${childPos.y}, ${childPos.x - 8} ${childPos.y}"
                    fill="none" stroke="${p2Color}" stroke-width="${isCrossPool ? 2 : 1.2}" opacity="${strokeOpacity}"
                    stroke-dasharray="${isCrossPool ? '4,3' : 'none'}"/>`;
            }
        }

        // Legend for cross-pool
        const crossCount = visiblePairs.filter(bp => bp.cross_pool).length;
        if (crossCount > 0) {
            svgContent += `<text x="${svgW / 2}" y="${svgH - 4}" text-anchor="middle" fill="#a855f7" font-size="10">
                ${crossCount} cross-pool breeding${crossCount > 1 ? 's' : ''} (dashed lines)</text>`;
        }

        svg.innerHTML = svgContent;

        // Update cross-pool stat in header
        const crossPoolStat = document.getElementById('dd-cross-pool-stat');
        if (crossPoolStat) {
            const totalCross = gen.cross_pool_breeding || 0;
            crossPoolStat.textContent = totalCross > 0
                ? `🔀 ${totalCross} cross-pool breed${totalCross > 1 ? 's' : ''}`
                : '';
        }
    }

    /* helpers for breeding flow */
    _getAgentPoolIndex(agentId) {
        if (!this.ddGenerations.length) return 0;
        for (const gen of this.ddGenerations) {
            const a = (gen.agents || []).find(ag => ag.agent_id === agentId);
            if (a) return a.pool_index || 0;
        }
        return 0;
    }

    _getAgentFitness(agentId) {
        for (const gen of this.ddGenerations) {
            const a = (gen.agents || []).find(ag => ag.agent_id === agentId);
            if (a && a.fitness !== null && a.fitness !== undefined && a.fitness > -999) {
                return parseFloat(a.fitness).toFixed(3);
            }
        }
        return '?';
    }

    /* ------------------------------------------------------------------
       Pool-to-Pool Breeding Matrix
       ------------------------------------------------------------------ */
    renderPoolMatrix() {
        const container = document.getElementById('dd-pool-matrix');
        if (!container) return;

        if (this.ddGenerations.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:20px;text-align:center;font-size:11px;">No data</div>';
            return;
        }

        // Aggregate pool breed matrix across all visible generations
        const aggregated = {};
        let maxPoolIdx = 0;
        for (const gen of this.ddGenerations) {
            const matrix = gen.pool_breed_matrix || {};
            for (const [key, count] of Object.entries(matrix)) {
                aggregated[key] = (aggregated[key] || 0) + count;
                const [a, b] = key.split('-').map(Number);
                maxPoolIdx = Math.max(maxPoolIdx, a, b);
            }
        }

        if (Object.keys(aggregated).length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:20px;text-align:center;font-size:11px;">No breeding data yet</div>';
            return;
        }

        const poolCount = Math.min(maxPoolIdx + 1, 8); // cap at 8 for readability
        let maxCount = Math.max(...Object.values(aggregated), 1);

        let html = '<table class="dd-matrix-table"><thead><tr><th></th>';
        for (let i = 0; i < poolCount; i++) {
            const color = DeepDiveManager.poolColor(i);
            html += `<th><span class="dd-matrix-pool-dot" style="background:${color};"></span>${i + 1}</th>`;
        }
        html += '</tr></thead><tbody>';

        for (let r = 0; r < poolCount; r++) {
            const rColor = DeepDiveManager.poolColor(r);
            html += `<tr><td><span class="dd-matrix-pool-dot" style="background:${rColor};"></span>${r + 1}</td>`;
            for (let c = 0; c < poolCount; c++) {
                const key = `${Math.min(r, c)}-${Math.max(r, c)}`;
                const count = aggregated[key] || 0;
                const intensity = count / maxCount;
                const bg = count > 0
                    ? (r === c
                        ? `rgba(74,222,128,${0.15 + intensity * 0.6})`
                        : `rgba(168,85,247,${0.15 + intensity * 0.6})`)
                    : 'transparent';
                html += `<td style="background:${bg};" title="Pool ${r + 1} × Pool ${c + 1}: ${count}">${count || ''}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        container.innerHTML = html;
    }

    /* ------------------------------------------------------------------
       Top Breeders — Most frequently selected parents
       ------------------------------------------------------------------ */
    renderTopBreeders(breeders) {
        const container = document.getElementById('dd-top-breeders');
        if (!container) return;

        if (!breeders || breeders.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:20px;text-align:center;font-size:11px;">No breeder data</div>';
            return;
        }

        container.innerHTML = breeders.slice(0, 10).map((b, i) => {
            const color = DeepDiveManager.poolColor(b.pool_index || 0);
            const idShort = (b.agent_id || '').substring(0, 14);
            const fit = (b.fitness !== null && b.fitness !== undefined && b.fitness > -999)
                ? parseFloat(b.fitness).toFixed(4) : '—';
            const barW = Math.max(10, (b.breed_count / (breeders[0]?.breed_count || 1)) * 100);
            return `<div class="dd-breeder-item" title="${b.agent_id}">
                <div class="dd-breeder-rank">${i + 1}</div>
                <div class="dd-breeder-swatch" style="background:${color};"></div>
                <div class="dd-breeder-info">
                    <span class="dd-breeder-id">${idShort}</span>
                    <span class="dd-breeder-fit">${fit}</span>
                </div>
                <div class="dd-breeder-bar-wrap">
                    <div class="dd-breeder-bar" style="width:${barW}%;background:${color};"></div>
                    <span class="dd-breeder-count">${b.breed_count}×</span>
                </div>
            </div>`;
        }).join('');
    }

    /* ------------------------------------------------------------------
       Method Direction Chart — update data
       ------------------------------------------------------------------ */
    updateMethodChart(methodDir) {
        if (!this.methodChart || !methodDir || methodDir.length === 0) return;

        this.methodChart.data.labels = methodDir.map(m => m.generation);
        this.methodChart.data.datasets[0].data = methodDir.map(m => m.avg_rle_min_run ?? null);
        this.methodChart.data.datasets[1].data = methodDir.map(m => m.avg_recipe_strength ?? 0);
        this.methodChart.data.datasets[2].data = methodDir.map(m => m.cross_pool_breeding || 0);
        this.methodChart.data.datasets[3].data = methodDir.map(m => m.novel_method_count || 0);
        this.methodChart.data.datasets[4].data = methodDir.map(m => m.unique_families || 0);
        this.methodChart.update('none');
    }

    /* ------------------------------------------------------------------
       Method Direction Summary Cards — Enhanced v0.9.10
       ------------------------------------------------------------------ */
    renderMethodSummary(methodDir) {
        const container = document.getElementById('dd-method-summary');
        if (!container) return;

        if (!methodDir || methodDir.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:20px;text-align:center;font-size:11px;">Waiting for data...</div>';
            return;
        }

        const latest = methodDir[methodDir.length - 1] || {};
        const first = methodDir[0] || {};

        // RLE trend
        const rleLatest = latest.avg_rle_min_run;
        const rleFirst = first.avg_rle_min_run;
        let rleTrend = '—';
        let rleTrendColor = 'var(--text-muted)';
        if (rleLatest != null && rleFirst != null) {
            const diff = rleLatest - rleFirst;
            rleTrend = diff > 0 ? `▲ ${diff.toFixed(2)}` : diff < 0 ? `▼ ${Math.abs(diff).toFixed(2)}` : '→ 0';
            rleTrendColor = diff > 0 ? '#f43f5e' : diff < 0 ? '#4ade80' : 'var(--text-muted)';
        }

        // Novel methods total
        const totalNovel = methodDir.reduce((s, m) => s + (m.novel_method_count || 0), 0);

        // Cross-pool total
        const totalCross = methodDir.reduce((s, m) => s + (m.cross_pool_breeding || 0), 0);

        // v0.9.10: Strength + family stats
        const avgStrength = (latest.avg_recipe_strength || 0).toFixed(2);
        const strengthPct = Math.round(parseFloat(avgStrength) * 100);
        const strengthColor = strengthPct >= 70 ? '#4ade80' : strengthPct >= 30 ? '#f59e0b' : '#f43f5e';
        const matureCount = latest.mature_recipe_count || 0;
        const uniqueFamilies = latest.unique_families || 0;
        const dominantFamily = latest.dominant_family || 'none';
        const resurrectedCount = latest.resurrected_count || 0;

        // v0.9.10: Method registry data (stored on the manager from poll)
        const reg = this._methodRegistry || {};
        const totalFamilies = reg.total_families || 0;
        const aliveFamilies = reg.alive_families || 0;
        const deadFamilies = reg.dead_families || 0;
        const totalRediscoveries = reg.total_rediscoveries || 0;

        container.innerHTML = `
            <div class="dd-method-card">
                <div class="dd-method-card-label">Avg Recipe Strength</div>
                <div class="dd-method-card-value" style="color:${strengthColor};">
                    <span style="display:inline-block;width:50px;height:8px;background:#333;border-radius:4px;vertical-align:middle;margin-right:6px;">
                        <span style="display:block;width:${strengthPct}%;height:100%;background:${strengthColor};border-radius:4px;"></span>
                    </span>${avgStrength}
                </div>
                <div class="dd-method-card-sub">LoRA-like intensity</div>
            </div>
            <div class="dd-method-card">
                <div class="dd-method-card-label">Novel Methods</div>
                <div class="dd-method-card-value" style="color:#f43f5e;">${matureCount} <small style="font-size:10px;opacity:.7">now</small></div>
                <div class="dd-method-card-sub">${totalNovel} total across ${methodDir.length} gen${methodDir.length > 1 ? 's' : ''}</div>
            </div>
            <div class="dd-method-card">
                <div class="dd-method-card-label">Method Families</div>
                <div class="dd-method-card-value" style="color:#10b981;">${uniqueFamilies} <small style="font-size:10px;opacity:.7">active</small></div>
                <div class="dd-method-card-sub">Dominant: ${dominantFamily}</div>
            </div>
            <div class="dd-method-card">
                <div class="dd-method-card-label">RLE Trend</div>
                <div class="dd-method-card-value" style="color:${rleTrendColor};">${rleTrend}</div>
                <div class="dd-method-card-sub">Current: ${rleLatest != null ? rleLatest.toFixed(2) : '—'}</div>
            </div>
            <div class="dd-method-card">
                <div class="dd-method-card-label">Cross-Pool Breeds</div>
                <div class="dd-method-card-value" style="color:#a855f7;">${totalCross}</div>
                <div class="dd-method-card-sub">genetic diversity events</div>
            </div>
            <div class="dd-method-card">
                <div class="dd-method-card-label">Method Registry</div>
                <div class="dd-method-card-value" style="color:#22d3ee;">${totalFamilies}</div>
                <div class="dd-method-card-sub">🟢${aliveFamilies} alive · 💀${deadFamilies} graveyard · 🧟${totalRediscoveries} resurrections</div>
            </div>
        `;
    }

    /* ------------------------------------------------------------------
       Generation Detail Cards (Breeding + Mutations) — Enhanced
       ------------------------------------------------------------------ */
    renderGenerationsDetail(generations) {
        const container = document.getElementById('dd-generations-container');
        if (!container) return;

        if (!generations || generations.length === 0) {
            container.innerHTML = '<div class="placeholder-cell" style="padding:24px;text-align:center;">Waiting for data...</div>';
            return;
        }

        const reversed = [...generations].reverse();

        let html = '';
        for (const gen of reversed) {
            const genNum = gen.generation || 0;
            const isExpanded = this.expandedDDGens.has(genNum);
            const expandedClass = isExpanded ? 'expanded' : '';
            const toggleChar = isExpanded ? '▾' : '▸';
            const bestFit = gen.best_fitness !== undefined ? parseFloat(gen.best_fitness).toFixed(4) : '—';
            const avgFit = gen.avg_fitness !== undefined ? parseFloat(gen.avg_fitness).toFixed(4) : '—';
            const minFit = gen.min_fitness !== undefined ? parseFloat(gen.min_fitness).toFixed(4) : '—';
            const crossovers = gen.crossover_count || 0;
            const mutations = gen.mutation_count || 0;
            const crossPool = gen.cross_pool_breeding || 0;
            const novelCount = gen.novel_method_count || 0;

            html += `<div class="dd-gen-card">
                <div class="dd-gen-card-header" onclick="window._ddMgr.toggleDDGen(${genNum})">
                    <span class="dd-gen-title"><span style="color:var(--primary-color);">${toggleChar}</span> Generation ${genNum}</span>
                    <div class="dd-gen-stats">
                        <span>Best: <span class="dd-gen-stat-val" style="color:#4ade80;">${bestFit}</span></span>
                        <span>Avg: <span class="dd-gen-stat-val">${avgFit}</span></span>
                        <span>Min: <span class="dd-gen-stat-val" style="color:#ef4444;">${minFit}</span></span>
                        <span>🧬 ${crossovers}</span>
                        <span>⚡ ${mutations}</span>
                        ${crossPool > 0 ? `<span title="Cross-pool breeds" style="color:#a855f7;">🔀 ${crossPool}</span>` : ''}
                        ${novelCount > 0 ? `<span title="Novel methods" style="color:#f43f5e;">🆕 ${novelCount}</span>` : ''}
                    </div>
                </div>
                <div class="dd-gen-body ${expandedClass}" id="dd-gen-body-${genNum}">`;

            // Breeding pairs — enriched with fitness comparison
            const pairs = gen.breeding_pairs || [];
            if (pairs.length > 0) {
                html += '<div style="margin-bottom:8px;font-size:12px;color:var(--text-muted);font-weight:600;">Breeding Pairs:</div>';
                html += '<div class="dd-breed-list">';
                for (const bp of pairs.slice(0, 30)) {
                    const p1Short = (bp.parent1 || '').substring(0, 10);
                    const p2Short = (bp.parent2 || '').substring(0, 10);
                    const childShort = (bp.child || '').substring(0, 10);
                    const p1Fit = bp.parent1_fitness != null ? parseFloat(bp.parent1_fitness).toFixed(3) : '?';
                    const p2Fit = bp.parent2_fitness != null ? parseFloat(bp.parent2_fitness).toFixed(3) : '?';
                    const chFit = bp.child_fitness != null ? parseFloat(bp.child_fitness).toFixed(3) : '?';
                    const p1Color = DeepDiveManager.poolColor(bp.parent1_pool || 0);
                    const p2Color = DeepDiveManager.poolColor(bp.parent2_pool || 0);
                    const chColor = DeepDiveManager.poolColor(bp.child_pool || 0);
                    const crossClass = bp.cross_pool ? ' dd-breed-cross-pool' : '';

                    // Outcome indicator: did child improve over parents?
                    const parentBest = Math.max(bp.parent1_fitness || -9999, bp.parent2_fitness || -9999);
                    const childFit = bp.child_fitness || -9999;
                    const outcomeIcon = childFit > parentBest ? '📈' : childFit === parentBest ? '➡️' : '📉';

                    html += `<div class="dd-breed-pair${crossClass}" title="P1: ${p1Fit} + P2: ${p2Fit} → Child: ${chFit}">
                        <span class="dd-breed-agent-dot" style="background:${p1Color};"></span>
                        <span style="color:#22d3ee;" title="${bp.parent1} (${p1Fit})">${p1Short}</span>
                        <span class="dd-breed-arrow">×</span>
                        <span class="dd-breed-agent-dot" style="background:${p2Color};"></span>
                        <span style="color:#f43f5e;" title="${bp.parent2} (${p2Fit})">${p2Short}</span>
                        <span class="dd-breed-arrow">${outcomeIcon}</span>
                        <span class="dd-breed-agent-dot" style="background:${chColor};"></span>
                        <span style="color:#4ade80;" title="${bp.child} (${chFit})">${childShort}</span>
                    </div>`;
                }
                if (pairs.length > 30) {
                    html += `<div class="dd-breed-pair" style="color:var(--text-muted);">+${pairs.length - 30} more</div>`;
                }
                html += '</div>';
            } else {
                html += '<div style="font-size:12px;color:var(--text-muted);padding:4px 0;">No breeding pairs recorded</div>';
            }

            // Agent fitness mini-bar chart (top 20 agents)
            const agents = gen.agents || [];
            if (agents.length > 0) {
                const topAgents = agents.slice(0, 20);
                const maxFitVal = Math.max(...topAgents.map(a => a.fitness || 0), 0.001);
                html += '<div style="margin-top:8px;font-size:12px;color:var(--text-muted);font-weight:600;">Top Agent Fitness:</div>';
                html += '<div style="display:flex;align-items:flex-end;gap:3px;height:60px;margin-top:4px;">';
                for (const a of topAgents) {
                    const fit = a.fitness || 0;
                    const pct = Math.max(3, (fit / maxFitVal) * 100);
                    const color = DeepDiveManager.poolColor(a.pool_index || 0);
                    const idShort = (a.agent_id || '').substring(0, 8);
                    html += `<div style="flex:1;min-width:8px;max-width:32px;height:${pct}%;background:${color};border-radius:3px 3px 0 0;cursor:pointer;" title="${idShort}: ${parseFloat(fit).toFixed(4)}"></div>`;
                }
                html += '</div>';
            }

            html += '</div></div>';
        }

        container.innerHTML = html;
    }

    /* ------------------------------------------------------------------
       Toggle Generation Detail
       ------------------------------------------------------------------ */
    toggleDDGen(genNum) {
        const body = document.getElementById(`dd-gen-body-${genNum}`);
        if (!body) return;
        if (this.expandedDDGens.has(genNum)) {
            this.expandedDDGens.delete(genNum);
            body.classList.remove('expanded');
        } else {
            this.expandedDDGens.add(genNum);
            body.classList.add('expanded');
        }
    }
}

// Instantiate after DOM is ready
const ddManager = new DeepDiveManager();
window._ddMgr = ddManager;
