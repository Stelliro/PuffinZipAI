/* ============================================================================
   Chart.js - Graph Management
   ============================================================================ */

let fitnessChart = null;

function initializeChart() {
    const ctx = document.getElementById('fitnessChart');
    if (!ctx) return;
    
    fitnessChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Fitness Score',
                    data: [],
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#6366f1',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    borderWidth: 2,
                },
                {
                    label: 'Compression Ratio',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.05)',
                    tension: 0.4,
                    fill: false,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#8b5cf6',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    borderWidth: 2,
                }
                ,
                {
                    label: 'Baseline (Best Zip)',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255,107,107,0.02)',
                    tension: 0,
                    fill: false,
                    pointRadius: 0,
                    borderDash: [6,4],
                    borderWidth: 2,
                    hidden: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#cbd5e1',
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12,
                            weight: 600
                        }
                    }
                },
                grid: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.05)',
                        drawBorder: false
                    }
                }
            }
        }
    });
}

function updateChart(metrics) {
    if (!fitnessChart) return;
    
    // Prepare data
    const labels = metrics.map(m => `Gen ${m.generation}`);
    const fitnessData = metrics.map(m => m.fitness);
    const compressionData = metrics.map(m => m.compression_ratio);
    
    // Update chart
    fitnessChart.data.labels = labels;
    fitnessChart.data.datasets[0].data = fitnessData;
    fitnessChart.data.datasets[1].data = compressionData;
    // Try to fetch validation baseline and apply it
    fetch('/api/validation').then(r => r.json()).then(val => {
        if (val && val.success && val.validation) {
            const v = val.validation;
            // compute baseline compression ratio if sizes available
            let baselineRatio = null;
            if (v.baseline_compressed_size && v.original_size) {
                baselineRatio = (v.original_size - v.baseline_compressed_size) / v.original_size;
            } else if (typeof v.baseline_compression_ratio !== 'undefined') {
                baselineRatio = v.baseline_compression_ratio;
            }
            if (baselineRatio !== null) {
                // Fill baseline dataset to match labels
                fitnessChart.data.datasets[2].data = labels.map(_ => baselineRatio);
                fitnessChart.data.datasets[2].hidden = false;
            }
        }

        // Adjust coloring and indicator based on latest point vs baseline
        try {
            const latestCompression = compressionData.length ? compressionData[compressionData.length - 1] : null;
            const baselineVal = fitnessChart.data.datasets[2].data.length ? fitnessChart.data.datasets[2].data[0] : null;
            updateChartColorsAndIndicator(latestCompression, baselineVal, fitnessData);
        } catch (e) {
            console.warn('updateChart: indicator update failed', e);
        }

        fitnessChart.update('none'); // Update without animation
    }).catch(() => {
        fitnessChart.update('none');
    });
}

function addMetricToChart(metric) {
    if (!fitnessChart) return;
    
    fitnessChart.data.labels.push(`Gen ${metric.generation}`);
    fitnessChart.data.datasets[0].data.push(metric.fitness);
    fitnessChart.data.datasets[1].data.push(metric.compression_ratio);
    
    // Keep only last 100 data points for performance
    if (fitnessChart.data.labels.length > 100) {
        fitnessChart.data.labels.shift();
        fitnessChart.data.datasets[0].data.shift();
        fitnessChart.data.datasets[1].data.shift();
    }
    
    fitnessChart.update('none');
    // update baseline and indicator after adding a point
    fetch('/api/validation').then(r => r.json()).then(val => {
        if (val && val.success && val.validation) {
            const v = val.validation;
            let baselineRatio = null;
            if (v.baseline_compressed_size && v.original_size) {
                baselineRatio = (v.original_size - v.baseline_compressed_size) / v.original_size;
            } else if (typeof v.baseline_compression_ratio !== 'undefined') {
                baselineRatio = v.baseline_compression_ratio;
            }
            if (baselineRatio !== null) {
                fitnessChart.data.datasets[2].data = fitnessChart.data.labels.map(_ => baselineRatio);
                fitnessChart.data.datasets[2].hidden = false;
            }
        }

        const latestCompression = fitnessChart.data.datasets[1].data.length ? fitnessChart.data.datasets[1].data[fitnessChart.data.datasets[1].data.length - 1] : null;
        const baselineVal = fitnessChart.data.datasets[2].data.length ? fitnessChart.data.datasets[2].data[0] : null;
        updateChartColorsAndIndicator(latestCompression, baselineVal, fitnessChart.data.datasets[0].data);
        fitnessChart.update('none');
    }).catch(() => {});
}

function updateChartColorsAndIndicator(latestCompression, baselineVal, fitnessSeries) {
    const indicatorEl = document.getElementById('graph-indicator');
    if (!indicatorEl) return;

    // default styles
    indicatorEl.className = '';
    indicatorEl.textContent = '';

    // Decide trend of fitness series
    let trendUp = false;
    if (fitnessSeries && fitnessSeries.length >= 2) {
        const last = fitnessSeries[fitnessSeries.length - 1];
        const prev = fitnessSeries[fitnessSeries.length - 2];
        trendUp = (last - prev) > 0;
    }

    // Compare latest compression (higher is better) to baseline
    if (baselineVal !== null && latestCompression !== null) {
        const delta = latestCompression - baselineVal;
        if (delta >= 0) {
            // We are at or above baseline: mark success
            // color fitness green
            fitnessChart.data.datasets[0].borderColor = '#10b981';
            fitnessChart.data.datasets[0].backgroundColor = 'rgba(16,185,129,0.08)';
            indicatorEl.classList.add('graph-indicator-success');
            indicatorEl.innerHTML = '✅ Above baseline (' + (delta * 100).toFixed(2) + '%)';
        } else {
            // below baseline: show how far to go and trend
            if (trendUp) {
                indicatorEl.classList.add('graph-indicator-warning');
                indicatorEl.innerHTML = '▲ Approaching baseline (' + (Math.abs(delta) * 100).toFixed(2) + '%)';
            } else {
                indicatorEl.classList.add('graph-indicator-danger');
                indicatorEl.innerHTML = '▼ Below baseline (' + (Math.abs(delta) * 100).toFixed(2) + '%)';
            }
            // color fitness line orange/red depending on distance
            if (Math.abs(delta) < 0.05) {
                fitnessChart.data.datasets[0].borderColor = '#f59e0b';
                fitnessChart.data.datasets[0].backgroundColor = 'rgba(245,158,11,0.06)';
            } else {
                fitnessChart.data.datasets[0].borderColor = '#ef4444';
                fitnessChart.data.datasets[0].backgroundColor = 'rgba(239,68,68,0.06)';
            }
        }
    } else {
        // No baseline available - indicate trend only
        if (trendUp) {
            indicatorEl.classList.add('graph-indicator-success');
            indicatorEl.innerHTML = '▲ Trend Up';
            fitnessChart.data.datasets[0].borderColor = '#6366f1';
            fitnessChart.data.datasets[0].backgroundColor = 'rgba(99, 102, 241, 0.1)';
        } else {
            indicatorEl.classList.add('graph-indicator-warning');
            indicatorEl.innerHTML = '▼ Trend Down';
            fitnessChart.data.datasets[0].borderColor = '#6366f1';
            fitnessChart.data.datasets[0].backgroundColor = 'rgba(99, 102, 241, 0.1)';
        }
    }
}

// Refresh chart button
document.addEventListener('DOMContentLoaded', function() {
    const refreshBtn = document.getElementById('graph-refresh');
    const fullscreenBtn = document.getElementById('graph-fullscreen');
    
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    if (data.metrics.length > 0) {
                        updateChart(data.metrics);
                    }
                });
        });
    }
    
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', function() {
            const graphSection = document.querySelector('.graph-section');
            if (graphSection.requestFullscreen) {
                graphSection.requestFullscreen();
            } else if (graphSection.webkitRequestFullscreen) {
                graphSection.webkitRequestFullscreen();
            }
        });
    }
});
