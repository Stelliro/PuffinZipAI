"""
Tests for the PuffinZipAI WebUI server — focused on generation lifecycle.

Tests cover:
- Basic route accessibility (index, status, metrics, logs, population, methods)
- Starting training via /api/training/start
- Metrics accumulation during generations
- Stopping training via /api/training/stop
- Infinite mode flag propagation
- System-limits exposure
- Edge cases (double start, stop when idle, etc.)
"""

import json
import threading
import time
import queue
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Helpers: fake optimizer that produces N generations ──────────────────────
class FakeEvolvingOptimizer:
    """Minimal stand-in for EvolutionaryOptimizer.
    
    When start_evolution() is called it pushes METRICS_JSON messages into the
    Bridge queue so the server accumulates metrics_history entries, then returns.
    """

    gui_output_queue: queue.Queue[str]

    def __init__(self, **kwargs):
        self.population_size = kwargs.get("population_size", 10)
        self.num_generations = kwargs.get("num_generations", 5)
        self.gui_output_queue = kwargs.get("gui_output_queue", queue.Queue())
        self.gui_stop_event = kwargs.get("gui_stop_event", threading.Event())
        self.infinite_mode = kwargs.get("infinite_mode", False)
        self.population_batch_size = kwargs.get("population_batch_size", 10)
        self.population = []  # empty but present

    def start_evolution(self):
        """Simulate a short evolution run, emitting metrics via the Bridge."""
        gens = self.num_generations if not self.infinite_mode else 3
        for g in range(1, gens + 1):
            if self.gui_stop_event.is_set():
                break
            payload = json.dumps({
                "generation": g,
                "fitness": round(0.5 + g * 0.1, 4),
                "ratio": round(50.0 + g * 2.0, 2),
                "benchmark_size": 1024 * g,
            })
            self.gui_output_queue.put_nowait(f"METRICS_JSON:{payload}")
            # Also emit a normal log line
            self.gui_output_queue.put_nowait(f"[ELS] Generation {g} complete")
            time.sleep(0.02)  # tiny delay to simulate work


# ── Patch the heavy imports *before* importing webui_server ─────────────────
# We mock puffinzip_ai imports so the server module loads cleanly without
# needing GPU drivers, numpy builds, etc.
_fake_puffinzip_ai = MagicMock()
_fake_puffinzip_ai.config = MagicMock()
_fake_puffinzip_ai.config.APP_VERSION = "TEST-0.0.1"
_fake_puffinzip_ai.config.ELS_LOG_PREFIX = "[ELS]"

_fake_hybrid = MagicMock()
_fake_registry = MagicMock()
_fake_registry.methods = {"rle_basic": True, "lz77_standard": True, "huffman_tree": True}
_fake_hybrid.registry = _fake_registry
_fake_get_hybrid_engine = MagicMock(return_value=_fake_hybrid)

# Patch sys.modules so `import puffinzip_ai` inside webui_server resolves
sys.modules.setdefault("puffinzip_ai", _fake_puffinzip_ai)
sys.modules.setdefault("puffinzip_ai.config", _fake_puffinzip_ai.config)
sys.modules.setdefault("puffinzip_ai.evolution_core", MagicMock())
sys.modules.setdefault("puffinzip_ai.evolution_core.evolutionary_optimizer", MagicMock())
sys.modules.setdefault("puffinzip_ai.hybrid_compression_engine", MagicMock())
sys.modules.setdefault("puffinzip_ai.logger", MagicMock())

# Now import the server — patch the two key objects it uses at module level
import webui_server
webui_server.EvolutionaryOptimizer = FakeEvolvingOptimizer
webui_server.get_hybrid_engine = _fake_get_hybrid_engine
webui_server.APP_VERSION = "TEST-0.0.1"
webui_server.ELS_LOG_PREFIX = "[ELS]"


# ═══════════════════════════════════════════════════════════════════════════════
# Test suite
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebuiGenerations(unittest.TestCase):
    """Integration tests for WebUI generation workflow."""

    # ── Setup / teardown per test ────────────────────────────────────────────
    def setUp(self):
        """Create a fresh Flask test client and reset global app_state."""
        webui_server.app.config["TESTING"] = True
        self.client = webui_server.app.test_client()
        # Reset the shared state so tests are independent
        webui_server.app_state = webui_server.AppState()

    # ── 1. Basic route smoke tests ───────────────────────────────────────────
    def test_index_returns_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"PuffinZipAI", resp.data)

    def test_status_returns_json(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("is_training", data)
        self.assertIn("current_generation", data)
        self.assertIn("system_limits", data)
        self.assertFalse(data["is_training"])

    def test_metrics_empty_when_idle(self):
        resp = self.client.get("/api/metrics")
        data = resp.get_json()
        self.assertEqual(data["generation"], 0)
        self.assertEqual(data["best_fitness"], 0.0)
        self.assertEqual(data["metrics"], [])

    def test_population_empty_when_idle(self):
        resp = self.client.get("/api/population")
        data = resp.get_json()
        self.assertEqual(data["population"], [])

    def test_logs_empty_when_idle(self):
        resp = self.client.get("/api/logs")
        data = resp.get_json()
        self.assertEqual(data, [])

    def test_compression_methods_returns_list(self):
        resp = self.client.get("/api/compression-methods")
        data = resp.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    # ── 2. System limits exposed via /api/status ─────────────────────────────
    def test_system_limits_keys(self):
        resp = self.client.get("/api/status")
        limits = resp.get_json()["system_limits"]
        for key in ("ram_gb", "cpu_cores", "default_pop", "default_gens", "max_pop", "max_gens"):
            self.assertIn(key, limits)

    # ── 3. Start training & generation metrics ───────────────────────────────
    def _start_and_wait(self, gens=5, pop=10, infinite=False, timeout=10):
        """Helper: POST /api/training/start and wait for thread to finish."""
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({
                "num_generations": gens,
                "population_size": pop,
                "infinite": infinite,
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])

        # Wait until training finishes (is_training goes False)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.client.get("/api/status").get_json()
            if not status["is_training"]:
                break
            time.sleep(0.05)
        else:
            self.fail("Training did not finish within timeout")

    def test_start_training_runs_generations(self):
        self._start_and_wait(gens=5, pop=10)

        metrics = self.client.get("/api/metrics").get_json()
        self.assertEqual(metrics["generation"], 5)
        self.assertGreater(metrics["best_fitness"], 0)
        self.assertEqual(len(metrics["metrics"]), 5)
        # Verify each generation recorded
        gen_numbers = [m["generation"] for m in metrics["metrics"]]
        self.assertEqual(gen_numbers, [1, 2, 3, 4, 5])

    def test_metrics_fitness_increases(self):
        """Our fake optimizer produces monotonically increasing fitness."""
        self._start_and_wait(gens=5, pop=10)
        metrics = self.client.get("/api/metrics").get_json()["metrics"]
        for i in range(1, len(metrics)):
            self.assertGreaterEqual(metrics[i]["fitness"], metrics[i - 1]["fitness"])

    def test_metrics_has_ratio_and_benchmark_size(self):
        self._start_and_wait(gens=3, pop=10)
        metrics = self.client.get("/api/metrics").get_json()["metrics"]
        for m in metrics:
            self.assertIn("ratio", m)
            self.assertIn("benchmark_size", m)
            self.assertGreater(m["ratio"], 0)
            self.assertGreater(m["benchmark_size"], 0)

    def test_benchmark_size_formatted_in_mb(self):
        self._start_and_wait(gens=2, pop=10)
        data = self.client.get("/api/metrics").get_json()
        # Should be a string ending in " MB"
        self.assertTrue(data["benchmark_size"].endswith("MB"))

    def test_history_pairs_match_metrics(self):
        self._start_and_wait(gens=4, pop=10)
        data = self.client.get("/api/metrics").get_json()
        self.assertEqual(len(data["history"]), len(data["metrics"]))
        for pair, m in zip(data["history"], data["metrics"]):
            self.assertEqual(pair[0], m["generation"])
            self.assertEqual(pair[1], m["fitness"])

    # ── 4. Logs generated during training ────────────────────────────────────
    def test_logs_produced_during_training(self):
        self._start_and_wait(gens=3, pop=10)
        # Drain all logs
        all_logs = []
        for _ in range(10):
            resp = self.client.get("/api/logs").get_json()
            all_logs.extend(resp)
            if not resp:
                break
        self.assertGreater(len(all_logs), 0)
        # Should contain at least the "Initializing" and "Finished" messages
        messages = " ".join(l["message"] for l in all_logs)
        self.assertIn("Initializing", messages)
        self.assertIn("Finished", messages)

    # ── 5. Stop training mid-run ─────────────────────────────────────────────
    def test_stop_training(self):
        # Start with many generations
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 500, "population_size": 10}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])

        # Give it a moment to start
        time.sleep(0.15)
        self.assertTrue(self.client.get("/api/status").get_json()["is_training"])

        # Send stop
        resp = self.client.post("/api/training/stop")
        self.assertTrue(resp.get_json()["success"])

        # Wait for thread to actually stop
        deadline = time.time() + 5
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)

        self.assertFalse(self.client.get("/api/status").get_json()["is_training"])
        # Fewer generations than requested
        metrics = self.client.get("/api/metrics").get_json()
        self.assertLess(metrics["generation"], 500)

    # ── 6. Double-start prevention ───────────────────────────────────────────
    def test_double_start_rejected(self):
        # First start succeeds
        r1 = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 200, "population_size": 10}),
            content_type="application/json",
        )
        self.assertTrue(r1.get_json()["success"])
        time.sleep(0.1)

        # Second start while first is running should fail
        r2 = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 5, "population_size": 10}),
            content_type="application/json",
        )
        self.assertFalse(r2.get_json()["success"])

        # Cleanup
        self.client.post("/api/training/stop")
        time.sleep(0.3)

    # ── 7. Stop when idle is safe ────────────────────────────────────────────
    def test_stop_when_idle(self):
        resp = self.client.post("/api/training/stop")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])

    # ── 8. Infinite mode flag ────────────────────────────────────────────────
    def test_infinite_mode(self):
        """In infinite mode the fake optimizer runs 3 gens then exits."""
        self._start_and_wait(gens=9999, pop=10, infinite=True)
        metrics = self.client.get("/api/metrics").get_json()
        # Fake optimizer in infinite mode produces exactly 3 generations
        self.assertEqual(metrics["generation"], 3)

    # ── 9. Population size clamped to system limits ──────────────────────────
    def test_population_clamped(self):
        """Request a huge population — server should clamp it."""
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 2, "population_size": 999999}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])
        # We can't directly inspect the arg, but it shouldn't crash
        deadline = time.time() + 5
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)
        self.assertFalse(self.client.get("/api/status").get_json()["is_training"])

    # ── 10. Generations clamped to system limits ─────────────────────────────
    def test_generations_clamped(self):
        """Request more generations than max — server should clamp."""
        max_gens = webui_server.SYSTEM_LIMITS["max_gens"]
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": max_gens + 5000, "population_size": 10}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])
        # Wait for finish
        deadline = time.time() + 30
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)
        metrics = self.client.get("/api/metrics").get_json()
        self.assertLessEqual(metrics["generation"], max_gens)

    # ── 11. Status evolution_time increases during training ──────────────────
    def test_evolution_time_increases(self):
        self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 200, "population_size": 10}),
            content_type="application/json",
        )
        time.sleep(0.15)
        t1 = self.client.get("/api/status").get_json().get("evolution_time", 0)
        time.sleep(0.2)
        t2 = self.client.get("/api/status").get_json().get("evolution_time", 0)
        self.assertGreater(t2, t1)
        # Cleanup
        self.client.post("/api/training/stop")
        time.sleep(0.3)

    # ── 12. Sequential training runs reset state ─────────────────────────────
    def test_sequential_runs_reset(self):
        """A second training run should start from gen 0, not continue."""
        self._start_and_wait(gens=3, pop=10)
        first_count = len(self.client.get("/api/metrics").get_json()["metrics"])
        self.assertEqual(first_count, 3)

        # Run again
        self._start_and_wait(gens=2, pop=10)
        second_data = self.client.get("/api/metrics").get_json()
        # The reset() clears metrics_history, so we should have exactly 2
        self.assertEqual(len(second_data["metrics"]), 2)
        self.assertEqual(second_data["generation"], 2)

    # ── 13. Batch size parameter accepted ─────────────────────────────────────
    def test_batch_size_accepted(self):
        """batch_size parameter should be accepted without error."""
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 3, "population_size": 20, "batch_size": 5}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])
        # Wait for finish
        deadline = time.time() + 10
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)
        self.assertFalse(self.client.get("/api/status").get_json()["is_training"])
        metrics = self.client.get("/api/metrics").get_json()
        self.assertEqual(metrics["generation"], 3)

    # ── 14. Batch size clamped to population ──────────────────────────────────
    def test_batch_size_clamped_to_pop(self):
        """batch_size > population should be clamped to population."""
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 2, "population_size": 10, "batch_size": 999}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])
        deadline = time.time() + 10
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)
        self.assertFalse(self.client.get("/api/status").get_json()["is_training"])

    # ── 15. Default batch size when not specified ─────────────────────────────
    def test_default_batch_size(self):
        """Not specifying batch_size should still work (defaults to 10)."""
        resp = self.client.post(
            "/api/training/start",
            data=json.dumps({"num_generations": 2, "population_size": 15}),
            content_type="application/json",
        )
        self.assertTrue(resp.get_json()["success"])
        deadline = time.time() + 10
        while time.time() < deadline:
            if not self.client.get("/api/status").get_json()["is_training"]:
                break
            time.sleep(0.05)
        self.assertFalse(self.client.get("/api/status").get_json()["is_training"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
