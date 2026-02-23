/* ============================================================================
   Chart.js - Global Helper Functions
   ============================================================================
   The main chart is created and managed by app.js (PuffinZipAIApp.initChart).
   This file only provides global helper functions for backwards compatibility.
   ============================================================================ */

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