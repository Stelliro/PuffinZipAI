/* ============================================================================
   Chart.js - Global Helper Functions
   ============================================================================
   The main chart is created and managed by app.js (PuffinZipAIApp.initChart).
   This file only provides global helper functions for backwards compatibility.
   ============================================================================ */

/** Archived chart runs — array of { labels, datasets[] } snapshots */
window._chartRunHistory = window._chartRunHistory || [];

window.addMetricToChart = function(metric) {
    // Delegate to the app's chart instance
    const chart = window.fitnessChart || (window.puffinApp && window.puffinApp.chart);
    if (!chart || !chart.data) return;

    const lastGen = chart.data.labels.length > 0 ? chart.data.labels[chart.data.labels.length - 1] : -1;
    if (metric.generation !== lastGen) {
        chart.data.labels.push(metric.generation);
        const ratio = metric.ratio !== undefined ? metric.ratio : 0;
        chart.data.datasets[0].data.push(parseFloat(ratio));
        const sizeKB = metric.benchmark_size ? (metric.benchmark_size / 1024).toFixed(1) : 0;
        chart.data.datasets[1].data.push(parseFloat(sizeKB));
        // Complexity value (3rd dataset)
        if (chart.data.datasets.length > 2) {
            chart.data.datasets[2].data.push(metric.complexity_value !== undefined ? metric.complexity_value : 0);
        }
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(ds => ds.data.shift());
        }
        chart.update('none');
    }
};

window.resetChart = function() {
    const chart = window.fitnessChart || (window.puffinApp && window.puffinApp.chart);
    if (chart) {
        chart.data.labels = [];
        chart.data.datasets.forEach(ds => ds.data = []);
        chart.update();
    }
};

/**
 * Archive the current chart data so it can be reviewed later, then clear
 * the chart for a fresh run.  Keeps up to 10 archived runs.
 */
window.archiveAndResetChart = function() {
    const chart = window.fitnessChart || (window.puffinApp && window.puffinApp.chart);
    if (!chart || !chart.data || chart.data.labels.length === 0) return;

    // Deep-copy current chart data into the archive
    const snapshot = {
        labels: [...chart.data.labels],
        datasets: chart.data.datasets.map(ds => ({
            label: ds.label,
            data: [...ds.data],
        })),
        archivedAt: new Date().toISOString(),
        runNumber: window._chartRunHistory.length + 1,
    };
    window._chartRunHistory.push(snapshot);
    // Cap at 10 archived runs
    if (window._chartRunHistory.length > 10) window._chartRunHistory.shift();

    console.log(`Chart run ${snapshot.runNumber} archived (${snapshot.labels.length} points).`);

    // Now reset the chart for the new run
    window.resetChart();
};