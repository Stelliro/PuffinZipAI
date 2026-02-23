# PuffinZipAI_Project/puffinzip_gui/generational_data_viewer.py
"""Generational Deep Dive viewer with collapsible generations / batches,
lazy-loading agent metrics, and pagination (20 generations per page).

Key behaviours
--------------
* **Collapsed by default** — generation rows only show summary stats.
* **Lazy load** — batch and agent rows are created only when expanded
  (``<<TreeviewOpen>>``), and destroyed on collapse (``<<TreeviewClose>>``).
  This keeps Tk memory usage constant regardless of history depth.
* **Pagination** — 20 generations per page; new generations auto-append to
  the last page when the user is already viewing it.
"""

import csv
import io
import json
import logging
import math
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Literal, TypeAlias

import numpy as np

# ── Fallback colours / fonts ────────────────────────────────────────────────
FRAME_BG_FALLBACK = "#333333"
LABEL_FG_FALLBACK = "#D4D4D4"
INPUT_BG_FALLBACK = "#252525"
INPUT_FG_FALLBACK = "#D4D4D4"
ACCENT_COLOR_FALLBACK = "#0078D4"
TREEVIEW_BG_FALLBACK = "#2A2A2A"
TREEVIEW_FG_FALLBACK = "#D0D0D0"
TREEVIEW_HEADING_BG_FALLBACK = "#3C3C3C"
TREEVIEW_HEADING_FG_FALLBACK = "#E0E0E0"
TREEVIEW_SELECTED_BG_FALLBACK = ACCENT_COLOR_FALLBACK
TREEVIEW_SELECTED_FG_FALLBACK = "#FFFFFF"

FONT_NORMAL_FALLBACK = ("Segoe UI", 10)
FONT_BOLD_FALLBACK = ("Segoe UI", 10, "bold")
FONT_SMALL_FALLBACK = ("Segoe UI", 9)
FONT_SECTION_TITLE_FALLBACK = ("Segoe UI", 12, "bold")

STYLE_PREFIX_GDV = "GenDataViewer."

# Glyph constants for tree labels
_GLYPH_GEN = "\U0001f9ec"     # 🧬
_GLYPH_BATCH = "\U0001f4e6"   # 📦
_GLYPH_AGENT = "\U0001f916"   # 🤖

GENS_PER_PAGE = 20
_DUMMY_SUFFIX = "__placeholder"

_Anchor: TypeAlias = Literal['nw', 'n', 'ne', 'w', 'center', 'e', 'sw', 's', 'se']


class GenerationalDataViewerTab(ttk.Frame):
    """Collapsible, lazily-loaded, paginated population viewer for ELS runs."""

    # ── Construction ────────────────────────────────────────────────────────
    def __init__(self, parent, app_instance, els_optimizer_instance, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app_instance
        self.logger: logging.Logger = getattr(app_instance, 'logger', None)  # type: ignore[assignment]
        if self.logger is None:
            self.logger = logging.getLogger("GenDataViewerFallbackLogger_PMA")
            if not self.logger.handlers:
                _h = logging.StreamHandler(sys.stdout)
                _h.setFormatter(logging.Formatter(
                    '%(asctime)s - GDV_FB - %(levelname)s - %(message)s'))
                self.logger.addHandler(_h)
                self.logger.setLevel(logging.INFO)
            self.logger.warning("GDV: Using fallback logger.")

        self.els_optimizer = els_optimizer_instance

        # --- Local snapshot store (lightweight dicts, no Q-tables) ---
        self.generation_snapshots: list = []
        self.current_page: int = 0
        self._lazy_loaded: set = set()      # iids whose real children are loaded
        self._was_on_last_page: bool = True  # auto-follow newest data

        # Keep compatibility field
        self.current_selected_generation_data: list = []

        # --- Theme colours ---
        self.frame_bg = getattr(self.app, 'FRAME_BG', FRAME_BG_FALLBACK)
        self.label_fg = getattr(self.app, 'LABEL_FG', LABEL_FG_FALLBACK)
        self.input_bg = getattr(self.app, 'INPUT_BG', INPUT_BG_FALLBACK)
        self.input_fg = getattr(self.app, 'INPUT_FG', INPUT_FG_FALLBACK)
        self.accent_color = getattr(self.app, 'ACCENT_COLOR', ACCENT_COLOR_FALLBACK)

        self.font_normal = getattr(self.app, 'FONT_NORMAL', FONT_NORMAL_FALLBACK)
        self.font_bold = getattr(self.app, 'FONT_BOLD', FONT_BOLD_FALLBACK)
        self.font_section_title = getattr(
            self.app, 'FONT_SECTION_TITLE', FONT_SECTION_TITLE_FALLBACK)
        self.font_small = getattr(self.app, 'FONT_SMALL', FONT_SMALL_FALLBACK)

        self.treeview_bg = getattr(self.app, 'TEXT_AREA_BG', TREEVIEW_BG_FALLBACK)
        self.treeview_fg = getattr(self.app, 'TEXT_AREA_FG', TREEVIEW_FG_FALLBACK)
        self.treeview_heading_bg = getattr(
            self.app, 'BUTTON_BG', TREEVIEW_HEADING_BG_FALLBACK)
        self.treeview_heading_fg = getattr(
            self.app, 'BUTTON_FG', TREEVIEW_HEADING_FG_FALLBACK)
        self.treeview_selected_bg = getattr(
            self.app, 'ACCENT_COLOR', TREEVIEW_SELECTED_BG_FALLBACK)
        self.treeview_selected_fg = getattr(
            self.app, 'TEXT_AREA_BG', TREEVIEW_SELECTED_FG_FALLBACK)

        self.configure(style=f"{STYLE_PREFIX_GDV}TFrame")
        self._setup_internal_styles()
        self._create_widgets()

        # Seed from optimizer if snapshots already exist (e.g. loaded checkpoint)
        self._sync_snapshots_from_optimizer()
        self.logger.info(
            "GenerationalDataViewerTab initialized (collapsible / paginated).")

    # ── Styles ──────────────────────────────────────────────────────────────
    def _setup_internal_styles(self):
        style = ttk.Style(self)
        style.configure(f"{STYLE_PREFIX_GDV}TFrame", background=self.frame_bg)
        style.configure(f"{STYLE_PREFIX_GDV}TLabel", background=self.frame_bg,
                        foreground=self.label_fg, font=self.font_normal)
        style.configure(f"{STYLE_PREFIX_GDV}Title.TLabel", background=self.frame_bg,
                        foreground=self.accent_color, font=self.font_section_title)
        style.configure(f"{STYLE_PREFIX_GDV}Small.TLabel", background=self.frame_bg,
                        foreground=self.label_fg, font=self.font_small)
        style.configure(f"{STYLE_PREFIX_GDV}TButton",
                        font=getattr(self.app, 'FONT_BUTTON', FONT_BOLD_FALLBACK))
        style.configure(f"{STYLE_PREFIX_GDV}TEntry",
                        fieldbackground=self.input_bg, foreground=self.input_fg,
                        insertbackground=self.label_fg, font=self.font_normal,
                        padding=4)
        style.map(f"{STYLE_PREFIX_GDV}TEntry",
                  bordercolor=[('focus', self.accent_color)])

        row_h = (int(self.font_small[1] * 2.2)
                 if self.font_small and len(self.font_small) > 1 else 25)
        style.configure(f"{STYLE_PREFIX_GDV}Treeview",
                        background=self.treeview_bg,
                        fieldbackground=self.treeview_bg,
                        foreground=self.treeview_fg,
                        font=self.font_small,
                        rowheight=row_h)
        style.configure(f"{STYLE_PREFIX_GDV}Treeview.Heading",
                        background=self.treeview_heading_bg,
                        foreground=self.treeview_heading_fg,
                        font=self.font_bold,
                        relief="flat", padding=(6, 6))
        style.map(f"{STYLE_PREFIX_GDV}Treeview.Heading",
                  background=[('active', self.accent_color),
                              ('!active', self.treeview_heading_bg)],
                  relief=[('active', 'groove')])
        style.map(f"{STYLE_PREFIX_GDV}Treeview",
                  background=[('selected', self.treeview_selected_bg)],
                  foreground=[('selected', self.treeview_selected_fg)])

    # ── Widget creation ─────────────────────────────────────────────────────
    def _create_widgets(self):
        # ── Top control bar ──
        controls = ttk.Frame(self, style=f"{STYLE_PREFIX_GDV}TFrame",
                             padding=(10, 8))
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="\U0001f504 Refresh",
                   style=f"{STYLE_PREFIX_GDV}TButton",
                   command=self._sync_and_refresh).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(controls, text="\U0001f4e5 Export Selected",
                   style=f"{STYLE_PREFIX_GDV}TButton",
                   command=self._export_selected).pack(side=tk.LEFT, padx=(0, 8))

        self.status_var = tk.StringVar(value="No data")
        ttk.Label(controls, textvariable=self.status_var,
                  style=f"{STYLE_PREFIX_GDV}Small.TLabel").pack(
            side=tk.LEFT, padx=(0, 16))

        # Pagination controls (right-aligned)
        page_frame = ttk.Frame(controls, style=f"{STYLE_PREFIX_GDV}TFrame")
        page_frame.pack(side=tk.RIGHT)

        self.prev_btn = ttk.Button(
            page_frame, text="\u25c0 Prev", width=8,
            style=f"{STYLE_PREFIX_GDV}TButton", command=self._prev_page)
        self.prev_btn.pack(side=tk.LEFT, padx=2)

        self.page_label_var = tk.StringVar(value="Page 1 / 1")
        ttk.Label(page_frame, textvariable=self.page_label_var,
                  style=f"{STYLE_PREFIX_GDV}TLabel").pack(side=tk.LEFT, padx=6)

        self.next_btn = ttk.Button(
            page_frame, text="Next \u25b6", width=8,
            style=f"{STYLE_PREFIX_GDV}TButton", command=self._next_page)
        self.next_btn.pack(side=tk.LEFT, padx=2)

        # ── Treeview ──
        tree_frame = ttk.Frame(self, style=f"{STYLE_PREFIX_GDV}TFrame",
                               padding=(10, 0, 10, 10))
        tree_frame.pack(fill=tk.BOTH, expand=True)

        col_keys = ("best_fit", "avg_fit", "agents", "gen_born",
                     "lr", "er", "rle_min", "thresholds")
        self.tree = ttk.Treeview(
            tree_frame, columns=col_keys,
            show="tree headings", selectmode="browse",
            style=f"{STYLE_PREFIX_GDV}Treeview")

        # Tree column #0 — name / label
        self.tree.heading("#0", text="Name", anchor=tk.W)
        self.tree.column("#0", width=300, minwidth=200, stretch=False)

        col_defs: dict[str, tuple[str, int, _Anchor]] = {
            "best_fit":   ("Best Fit",    90,  "e"),
            "avg_fit":    ("Avg Fit",     90,  "e"),
            "agents":     ("Agents",      60,  "center"),
            "gen_born":   ("Gen Born",    70,  "center"),
            "lr":         ("Learn Rate",  95,  "e"),
            "er":         ("Expl. Rate",  90,  "e"),
            "rle_min":    ("RLE MinRun",  85,  "center"),
            "thresholds": ("Thresholds",  180, "w"),
        }
        for key, (heading, width, anchor) in col_defs.items():
            self.tree.heading(key, text=heading, anchor=tk.CENTER)
            self.tree.column(key, width=width, anchor=anchor, stretch=tk.YES)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.tree.yview,
                             style="Vertical.TScrollbar")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                             command=self.tree.xview,
                             style="Horizontal.TScrollbar")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Events
        self.tree.bind("<<TreeviewOpen>>", self._on_expand)
        self.tree.bind("<<TreeviewClose>>", self._on_collapse)
        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Button-3>", self._on_right_click)

        # Right-click context menu
        self._ctx_menu = tk.Menu(self.tree, tearoff=0)

        # Tag colours
        error_color = getattr(self.app, 'ERROR_FG_COLOR', 'red')
        placeholder_color = getattr(self.app, 'DISABLED_FG_COLOR', 'grey')
        self.tree.tag_configure('gen_row', foreground=self.accent_color,
                                font=self.font_bold)
        self.tree.tag_configure('batch_row', foreground=self.label_fg,
                                font=self.font_normal)
        self.tree.tag_configure('agent_row', font=self.font_small)
        self.tree.tag_configure('error_row', foreground=error_color,
                                font=self.font_small)
        self.tree.tag_configure('placeholder_row', foreground=placeholder_color,
                                font=self.font_small)

        self.logger.debug("GDV widgets created (collapsible tree + pagination).")

    # ════════════════════════════════════════════════════════════════════════
    #  Public API (called from queue handler / primary_main_app)
    # ════════════════════════════════════════════════════════════════════════
    def on_generation_snapshot(self, gen_num: int):
        """Called by the queue handler when a new ``GEN_SNAPSHOT`` arrives."""
        if not self.els_optimizer or not hasattr(
                self.els_optimizer, 'generation_snapshots'):
            return

        existing_gens = {s['generation'] for s in self.generation_snapshots}
        for snap in self.els_optimizer.generation_snapshots:
            if snap['generation'] not in existing_gens:
                self.generation_snapshots.append(snap)
                existing_gens.add(snap['generation'])

        total_pages = self._total_pages()
        if self._was_on_last_page:
            self.current_page = total_pages - 1
            self._populate_page()
        else:
            # Just update controls so user sees page count change
            self._update_page_controls()

    def load_and_display_data(self):
        """Backward-compatible entry point — full refresh from optimizer."""
        self._sync_and_refresh()

    # ════════════════════════════════════════════════════════════════════════
    #  Snapshot synchronisation
    # ════════════════════════════════════════════════════════════════════════
    def _sync_snapshots_from_optimizer(self):
        """Pull all generation snapshots from the optimizer instance."""
        if self.els_optimizer and hasattr(
                self.els_optimizer, 'generation_snapshots'):
            existing_gens = {s['generation'] for s in self.generation_snapshots}
            for snap in self.els_optimizer.generation_snapshots:
                if snap['generation'] not in existing_gens:
                    self.generation_snapshots.append(snap)
                    existing_gens.add(snap['generation'])

    def _sync_and_refresh(self):
        """Sync snapshots from optimizer and repopulate the current page."""
        self._sync_snapshots_from_optimizer()

        # If there are no snapshots yet, try to create one from live population
        if not self.generation_snapshots:
            self._build_snapshot_from_live_population()

        total_pages = self._total_pages()
        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)
        self._populate_page()

    def _build_snapshot_from_live_population(self):
        """Fallback: create a single snapshot from the optimizer's current
        live population (useful before the first generation completes or
        when the optimizer has no snapshot support yet)."""
        if (not self.els_optimizer
                or not getattr(self.els_optimizer, 'population', None)):
            return

        population = self.els_optimizer.population
        gen_num = (getattr(self.els_optimizer,
                           'total_generations_elapsed', 0) or 0)

        # Avoid duplicating an already-captured generation
        if any(s['generation'] == gen_num for s in self.generation_snapshots):
            return

        batch_size = getattr(
            self.els_optimizer, '_agent_batch_size', len(population))
        if batch_size < 1:
            batch_size = len(population)

        batches_data = []
        for i in range(0, len(population), batch_size):
            chunk = population[i:i + batch_size]
            batch_agents = []
            batch_fits = []
            for agent in chunk:
                fit = (agent.get_fitness()
                       if hasattr(agent, 'get_fitness') else None)
                ai_core = (agent.get_puffin_ai()
                           if hasattr(agent, 'get_puffin_ai') else None)
                batch_agents.append({
                    "agent_id": getattr(agent, 'agent_id', 'N/A'),
                    "fitness": fit if fit is not None else float('nan'),
                    "generation_born": getattr(agent, 'generation_born', 0),
                    "parent_ids": list(getattr(agent, 'parent_ids', [])),
                    "learning_rate": (getattr(ai_core, 'learning_rate', 0.0)
                                      if ai_core else 0.0),
                    "exploration_rate": (
                        getattr(ai_core, 'exploration_rate', 0.0)
                        if ai_core else 0.0),
                    "rle_min_run": (
                        getattr(ai_core, 'rle_min_encodable_run_length', 'N/A')
                        if ai_core else 'N/A'),
                    "thresholds_str": (
                        ", ".join(map(str, ai_core.len_thresholds))
                        if ai_core and hasattr(ai_core, 'len_thresholds')
                        and ai_core.len_thresholds else "N/A"),
                    "evaluation_stats": dict(
                        getattr(agent, 'evaluation_stats', {}) or {})
                })
                if fit is not None and fit > -999:
                    batch_fits.append(fit)

            batches_data.append({
                "batch_idx": len(batches_data),
                "agents": batch_agents,
                "best_fitness": max(batch_fits) if batch_fits else 0.0,
                "avg_fitness": ((sum(batch_fits) / len(batch_fits))
                                if batch_fits else 0.0),
            })

        all_fits = [
            a.get_fitness() for a in population
            if (hasattr(a, 'get_fitness') and a.get_fitness() is not None
                and a.get_fitness() > -999)
        ]

        # Compute avg_fitness excluding catastrophic failures (eval fail/timeout)
        _OUTLIER_THRESHOLD = -50.0
        meaningful_fits = [f for f in all_fits if f > _OUTLIER_THRESHOLD]
        avg_fit = ((sum(meaningful_fits) / len(meaningful_fits))
                   if meaningful_fits else
                   ((sum(all_fits) / len(all_fits)) if all_fits else 0.0))

        import time as _t
        snap = {
            "generation": gen_num,
            "timestamp": _t.time(),
            "best_fitness": max(all_fits) if all_fits else 0.0,
            "avg_fitness": avg_fit,
            "min_fitness": min(all_fits) if all_fits else 0.0,
            "agent_count": len(population),
            "batch_count": len(batches_data),
            "batches": batches_data,
        }
        self.generation_snapshots.append(snap)

    # ════════════════════════════════════════════════════════════════════════
    #  Pagination
    # ════════════════════════════════════════════════════════════════════════
    def _total_pages(self) -> int:
        return max(1, math.ceil(
            len(self.generation_snapshots) / GENS_PER_PAGE))

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._was_on_last_page = False
            self._populate_page()

    def _next_page(self):
        if self.current_page < self._total_pages() - 1:
            self.current_page += 1
            self._was_on_last_page = (
                self.current_page >= self._total_pages() - 1)
            self._populate_page()

    def _update_page_controls(self):
        total = self._total_pages()
        self.page_label_var.set(f"Page {self.current_page + 1} / {total}")
        try:
            self.prev_btn.state(
                ['!disabled'] if self.current_page > 0 else ['disabled'])
            self.next_btn.state(
                ['!disabled'] if self.current_page < total - 1
                else ['disabled'])
        except Exception:
            pass  # ttk state may not be available in all themes
        n = len(self.generation_snapshots)
        self.status_var.set(
            f"{n} generation{'s' if n != 1 else ''} captured")

    # ════════════════════════════════════════════════════════════════════════
    #  Page population (only generation headers — no agents loaded)
    # ════════════════════════════════════════════════════════════════════════
    def _populate_page(self):
        """Clear tree and insert *collapsed* generation headers for the
        current page.  No batch or agent data is loaded yet."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._lazy_loaded.clear()

        total_gens = len(self.generation_snapshots)
        total_pages = self._total_pages()
        self.current_page = max(0, min(self.current_page, total_pages - 1))

        start_idx = self.current_page * GENS_PER_PAGE
        end_idx = min(start_idx + GENS_PER_PAGE, total_gens)
        page_snaps = self.generation_snapshots[start_idx:end_idx]

        if not page_snaps:
            self.tree.insert(
                "", tk.END,
                text="No generation data available.",
                values=("", "", "", "", "", "", "", ""),
                tags=('placeholder_row',))
            self._update_page_controls()
            return

        for snap in page_snaps:
            gen = snap['generation']
            iid = f"gen_{gen}"
            best = snap.get('best_fitness', 0.0)
            avg = snap.get('avg_fitness', 0.0)
            count = snap.get('agent_count', 0)
            n_batches = snap.get('batch_count', 0)

            label = (f"{_GLYPH_GEN} Generation {gen}  "
                     f"({count} agents, {n_batches} "
                     f"batch{'es' if n_batches != 1 else ''})")

            self.tree.insert(
                "", tk.END, iid=iid, text=label, open=False,
                values=(f"{best:.4f}", f"{avg:.4f}", str(count),
                        "", "", "", "", ""),
                tags=('gen_row',))

            # Dummy child → shows the expand arrow
            self.tree.insert(
                iid, tk.END, iid=f"{iid}{_DUMMY_SUFFIX}",
                text="  Loading\u2026",
                values=("", "", "", "", "", "", "", ""),
                tags=('placeholder_row',))

        self._update_page_controls()
        self.logger.debug(
            f"GDV: Page {self.current_page + 1}/{total_pages} populated "
            f"({len(page_snaps)} gen headers).")

    # ════════════════════════════════════════════════════════════════════════
    #  Lazy loading / unloading
    # ════════════════════════════════════════════════════════════════════════
    def _on_expand(self, event):
        """Load children lazily on first expand."""
        iid = self.tree.focus()
        if not iid or iid in self._lazy_loaded:
            return

        if iid.startswith("gen_") and "_batch_" not in iid:
            self._load_batches(iid)
        elif "_batch_" in iid and "_a_" not in iid:
            self._load_agents(iid)

        self._lazy_loaded.add(iid)

    def _on_collapse(self, event):
        """Destroy children on collapse to free Tk widget memory;
        re-insert a dummy so the expand arrow persists."""
        iid = self.tree.focus()
        if not iid:
            return

        # Delete all children recursively
        for child in self.tree.get_children(iid):
            self._delete_subtree(child)

        # Re-insert dummy
        self.tree.insert(
            iid, tk.END, iid=f"{iid}{_DUMMY_SUFFIX}",
            text="  Loading\u2026",
            values=("", "", "", "", "", "", "", ""),
            tags=('placeholder_row',))

        # Mark this and all descendants as not-loaded
        self._lazy_loaded.discard(iid)
        to_remove = [k for k in self._lazy_loaded
                     if k.startswith(iid + "_")]
        for k in to_remove:
            self._lazy_loaded.discard(k)

    def _delete_subtree(self, iid):
        """Recursively delete a tree node and all its children."""
        for child in self.tree.get_children(iid):
            self._delete_subtree(child)
        try:
            self.tree.delete(iid)
        except tk.TclError:
            pass

    # ── Load batch rows under a generation ──────────────────────────────────
    def _load_batches(self, gen_iid: str):
        gen_num = int(gen_iid.split("_")[1])
        snap = self._get_snapshot(gen_num)
        if not snap:
            return

        self._clear_children(gen_iid)

        batches = snap.get('batches', [])
        if not batches:
            self.tree.insert(
                gen_iid, tk.END,
                text="  No batch data recorded.",
                values=("", "", "", "", "", "", "", ""),
                tags=('placeholder_row',))
            return

        for batch in batches:
            b_idx = batch['batch_idx']
            b_iid = f"{gen_iid}_batch_{b_idx}"
            b_best = batch.get('best_fitness', 0.0)
            b_avg = batch.get('avg_fitness', 0.0)
            b_count = len(batch.get('agents', []))

            label = (f"{_GLYPH_BATCH} Batch {b_idx + 1}/{len(batches)}  "
                     f"({b_count} agents)")
            self.tree.insert(
                gen_iid, tk.END, iid=b_iid, text=label, open=False,
                values=(f"{b_best:.4f}", f"{b_avg:.4f}", str(b_count),
                        "", "", "", "", ""),
                tags=('batch_row',))

            # Dummy child for expand arrow
            self.tree.insert(
                b_iid, tk.END,
                iid=f"{b_iid}{_DUMMY_SUFFIX}",
                text="  Loading\u2026",
                values=("", "", "", "", "", "", "", ""),
                tags=('placeholder_row',))

    # ── Load agent rows under a batch ───────────────────────────────────────
    def _load_agents(self, batch_iid: str):
        # gen_<N>_batch_<M>
        parts = batch_iid.split("_")
        gen_num = int(parts[1])
        batch_idx = int(parts[3])

        snap = self._get_snapshot(gen_num)
        if not snap:
            return

        self._clear_children(batch_iid)

        batches = snap.get('batches', [])
        if batch_idx >= len(batches):
            return

        agents = batches[batch_idx].get('agents', [])
        for idx, agent_data in enumerate(agents):
            aid = agent_data.get('agent_id', 'N/A')
            display_id = aid if len(aid) <= 22 else "\u2026" + aid[-17:]
            fit = agent_data.get('fitness', float('nan'))

            # Safe iid: Tk forbids certain chars; also ensure uniqueness
            safe_aid = (aid.replace(" ", "_").replace(".", "_")
                        .replace("(", "").replace(")", ""))
            a_iid = f"{batch_iid}_a_{safe_aid}_{idx}"

            fit_str = (f"{fit:.4f}"
                       if isinstance(fit, (int, float)) and np.isfinite(fit)
                       else "N/A")
            lr = agent_data.get('learning_rate', 0.0)
            er = agent_data.get('exploration_rate', 0.0)
            rle_min = agent_data.get('rle_min_run', 'N/A')
            thresh = agent_data.get('thresholds_str', 'N/A')
            gen_born = agent_data.get('generation_born', '')

            self.tree.insert(
                batch_iid, tk.END, iid=a_iid,
                text=f"{_GLYPH_AGENT} {display_id}",
                values=(fit_str, "", "", str(gen_born),
                        f"{lr:.5f}", f"{er:.4f}",
                        str(rle_min), thresh),
                tags=('agent_row',))

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _clear_children(self, parent_iid: str):
        for child in self.tree.get_children(parent_iid):
            try:
                self.tree.delete(child)
            except tk.TclError:
                pass

    def _get_snapshot(self, gen_num: int):
        for snap in self.generation_snapshots:
            if snap['generation'] == gen_num:
                return snap
        return None

    # ════════════════════════════════════════════════════════════════════════
    #  Double-click detail popup
    # ════════════════════════════════════════════════════════════════════════
    def _on_double_click(self, event):
        iid = self.tree.focus()
        if not iid:
            return

        # Only show details for agent rows
        if "_a_" not in iid:
            return

        # Parse gen and batch from iid
        # Format: gen_<N>_batch_<M>_a_<safe_agent_id>_<idx>
        try:
            prefix, rest = iid.split("_a_", 1)
            prefix_parts = prefix.split("_")
            gen_num = int(prefix_parts[1])
            batch_idx = int(prefix_parts[3])
            # idx is the last segment after last _
            agent_list_idx = int(rest.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return

        snap = self._get_snapshot(gen_num)
        if not snap:
            return

        batches = snap.get('batches', [])
        if batch_idx >= len(batches):
            return

        agents = batches[batch_idx].get('agents', [])
        if agent_list_idx >= len(agents):
            return

        self._show_agent_detail_popup(agents[agent_list_idx])

    def _show_agent_detail_popup(self, agent_data: dict):
        lines = [
            f"Agent Details (ID: {agent_data.get('agent_id', 'N/A')})",
            "-" * 40,
        ]
        fit = agent_data.get('fitness', float('nan'))
        fit_ok = isinstance(fit, (int, float)) and np.isfinite(fit)
        lines.append(
            f"  Fitness: {fit:.4f}" if fit_ok else "  Fitness: N/A")
        lines.append(
            f"  Generation Born: {agent_data.get('generation_born', 'N/A')}")

        parent_ids = agent_data.get('parent_ids', [])
        lines.append(
            f"  Parents: "
            f"{', '.join(parent_ids) if parent_ids else 'N/A (Root/Immigrant)'}")

        lines.append("\n  Hyperparameters:")
        lines.append(
            f"    Learning Rate:    {agent_data.get('learning_rate', 0):.5f}")
        lines.append(
            f"    Exploration Rate: {agent_data.get('exploration_rate', 0):.4f}")
        lines.append(
            f"    RLE Min Run:      {agent_data.get('rle_min_run', 'N/A')}")
        lines.append(
            f"    Thresholds:       [{agent_data.get('thresholds_str', 'N/A')}]")

        eval_stats = agent_data.get('evaluation_stats', {})
        if eval_stats:
            lines.append("\n  Evaluation Stats:")
            stat_items = [
                ("Total Reward",       "total_reward",              "{:.4f}"),
                ("Items Evaluated",    "items_evaluated",           "{}"),
                ("Successful RLE",     "successful_rle",            "{}"),
                ("RLE Expansions",     "rle_expansion",             "{}"),
                ("NoCompression",      "chose_nocompression",       "{}"),
                ("AdvancedRLE",        "chose_adv_rle",             "{}"),
                ("Decomp Mismatches",  "decomp_failures_mismatch",  "{}"),
                ("RLE Errors",         "rle_errors_returned",       "{}"),
            ]
            for label, key, fmt in stat_items:
                val = eval_stats.get(key)
                if val is not None:
                    lines.append(f"    {label}: {fmt.format(val)}")
            proc_time = eval_stats.get('total_processing_time_ms')
            items_eval = eval_stats.get('items_evaluated', 0)
            if proc_time is not None and items_eval and items_eval > 0:
                lines.append(
                    f"    Avg Proc Time: {proc_time / items_eval:.2f} ms/item")
            novelty = eval_stats.get('novelty_adjustment')
            if novelty is not None:
                lines.append(f"    Novelty Adj:   {novelty:.4f}")
            gen_pen = eval_stats.get('gen_repetition_penalty')
            if gen_pen is not None:
                lines.append(f"    Gen Penalty:   {gen_pen:.4f}")
        else:
            lines.append("\n  Evaluation Stats: Not available")

        messagebox.showinfo("Agent Details", "\n".join(lines), parent=self)

        if hasattr(self.app, 'log_message'):
            aid = agent_data.get('agent_id', '?')
            fit_s = f"{fit:.4f}" if fit_ok else "N/A"
            self.app.log_message(
                f"GDV Detail: {aid} fit={fit_s}", "info_no_prefix")

    # ════════════════════════════════════════════════════════════════════════
    #  Right-click context menu & export
    # ════════════════════════════════════════════════════════════════════════
    def _on_right_click(self, event):
        """Show context menu with export options for the row under cursor."""
        iid = self.tree.identify_row(event.y)
        if not iid:
            return

        # Select the row under the cursor
        self.tree.selection_set(iid)
        self.tree.focus(iid)

        menu = self._ctx_menu
        menu.delete(0, tk.END)

        if iid.startswith("gen_") and "_batch_" not in iid:
            gen_num = int(iid.split("_")[1])
            menu.add_command(
                label=f"\U0001f4c4 Export Generation {gen_num} as CSV\u2026",
                command=lambda g=gen_num: self._export_generation(g, "csv"))
            menu.add_command(
                label=f"\U0001f4cb Export Generation {gen_num} as JSON\u2026",
                command=lambda g=gen_num: self._export_generation(g, "json"))
        elif "_batch_" in iid and "_a_" not in iid:
            parts = iid.split("_")
            gen_num = int(parts[1])
            batch_idx = int(parts[3])
            menu.add_command(
                label=f"\U0001f4c4 Export Batch {batch_idx + 1} as CSV\u2026",
                command=lambda g=gen_num, b=batch_idx: self._export_batch(
                    g, b, "csv"))
            menu.add_command(
                label=f"\U0001f4cb Export Batch {batch_idx + 1} as JSON\u2026",
                command=lambda g=gen_num, b=batch_idx: self._export_batch(
                    g, b, "json"))
        elif "_a_" in iid:
            # Agent row — offer export of the parent batch
            try:
                prefix = iid.split("_a_")[0]
                parts = prefix.split("_")
                gen_num = int(parts[1])
                batch_idx = int(parts[3])
                menu.add_command(
                    label=(f"\U0001f4c4 Export Parent Batch "
                           f"{batch_idx + 1} as CSV\u2026"),
                    command=lambda g=gen_num, b=batch_idx: self._export_batch(
                        g, b, "csv"))
                menu.add_command(
                    label=(f"\U0001f4cb Export Parent Batch "
                           f"{batch_idx + 1} as JSON\u2026"),
                    command=lambda g=gen_num, b=batch_idx: self._export_batch(
                        g, b, "json"))
            except (IndexError, ValueError):
                return
        else:
            return

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _export_selected(self):
        """Toolbar button: export whatever row is currently selected."""
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(
                "Export",
                "Select a generation or batch row first, then click Export.",
                parent=self)
            return

        iid = sel[0]
        if iid.startswith("gen_") and "_batch_" not in iid:
            gen_num = int(iid.split("_")[1])
            self._export_generation(gen_num, "csv")
        elif "_batch_" in iid and "_a_" not in iid:
            parts = iid.split("_")
            gen_num = int(parts[1])
            batch_idx = int(parts[3])
            self._export_batch(gen_num, batch_idx, "csv")
        elif "_a_" in iid:
            try:
                prefix = iid.split("_a_")[0]
                parts = prefix.split("_")
                gen_num = int(parts[1])
                batch_idx = int(parts[3])
                self._export_batch(gen_num, batch_idx, "csv")
            except (IndexError, ValueError):
                pass
        else:
            messagebox.showinfo(
                "Export",
                "Select a generation or batch row to export.",
                parent=self)

    # ── Export helpers ──────────────────────────────────────────────────────
    _CSV_COLUMNS = [
        "agent_id", "fitness", "generation_born", "learning_rate",
        "exploration_rate", "rle_min_run", "thresholds_str",
        "total_reward", "items_evaluated", "successful_rle",
        "rle_expansion", "chose_nocompression", "chose_adv_rle",
        "decomp_failures_mismatch", "rle_errors_returned",
        "total_original_bytes", "total_compressed_bytes",
        "total_processing_time_ms",
    ]

    def _flatten_agent(self, agent_data: dict) -> dict:
        """Convert a snapshot agent dict to a flat row for CSV export."""
        row: dict = {}
        for key in ("agent_id", "fitness", "generation_born",
                     "learning_rate", "exploration_rate",
                     "rle_min_run", "thresholds_str"):
            row[key] = agent_data.get(key, "")
        es = agent_data.get("evaluation_stats", {}) or {}
        for key in ("total_reward", "items_evaluated", "successful_rle",
                     "rle_expansion", "chose_nocompression", "chose_adv_rle",
                     "decomp_failures_mismatch", "rle_errors_returned",
                     "total_original_bytes", "total_compressed_bytes",
                     "total_processing_time_ms"):
            row[key] = es.get(key, "")
        return row

    def _export_generation(self, gen_num: int, fmt: str):
        """Export all agents in a generation to CSV or JSON."""
        snap = self._get_snapshot(gen_num)
        if not snap:
            messagebox.showerror(
                "Export Error",
                f"No snapshot data for generation {gen_num}.",
                parent=self)
            return

        default_name = f"generation_{gen_num}"
        batches = snap.get("batches", [])
        all_agents: list[dict] = []
        for b in batches:
            all_agents.extend(b.get("agents", []))

        if not all_agents:
            messagebox.showinfo(
                "Export",
                f"Generation {gen_num} has no agent data to export.",
                parent=self)
            return

        self._write_export(all_agents, default_name, fmt,
                           title=f"Export Generation {gen_num}")

    def _export_batch(self, gen_num: int, batch_idx: int, fmt: str):
        """Export all agents in a specific batch to CSV or JSON."""
        snap = self._get_snapshot(gen_num)
        if not snap:
            messagebox.showerror(
                "Export Error",
                f"No snapshot data for generation {gen_num}.",
                parent=self)
            return

        batches = snap.get("batches", [])
        if batch_idx >= len(batches):
            messagebox.showerror(
                "Export Error",
                f"Batch index {batch_idx} out of range.", parent=self)
            return

        agents = batches[batch_idx].get("agents", [])
        if not agents:
            messagebox.showinfo(
                "Export",
                f"Batch {batch_idx + 1} has no agent data to export.",
                parent=self)
            return

        default_name = f"gen{gen_num}_batch{batch_idx + 1}"
        self._write_export(agents, default_name, fmt,
                           title=f"Export Gen {gen_num} Batch {batch_idx + 1}")

    def _write_export(self, agents: list[dict], default_name: str,
                      fmt: str, title: str = "Export Data"):
        """Prompt for a save path and write agent data as CSV or JSON."""
        if fmt == "json":
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
            ext = ".json"
        else:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
            ext = ".csv"

        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=default_name + ext,
            parent=self)

        if not path:
            return  # user cancelled

        try:
            if fmt == "json":
                self._write_json(agents, path)
            else:
                self._write_csv(agents, path)

            count = len(agents)
            self.logger.info(
                f"GDV: Exported {count} agent(s) to {path} ({fmt})")
            if hasattr(self.app, 'log_message'):
                self.app.log_message(
                    f"Exported {count} agent(s) to {os.path.basename(path)}",
                    "info_no_prefix")
            messagebox.showinfo(
                "Export Complete",
                f"Exported {count} agent(s) to:\n{path}",
                parent=self)
        except Exception as exc:
            self.logger.error(f"GDV export error: {exc}", exc_info=True)
            messagebox.showerror(
                "Export Error", f"Failed to write file:\n{exc}",
                parent=self)

    def _write_csv(self, agents: list[dict], path: str):
        """Write agent data to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            for agent in agents:
                writer.writerow(self._flatten_agent(agent))

    def _write_json(self, agents: list[dict], path: str):
        """Write agent data to a pretty-printed JSON file."""
        # Export the full agent dicts (including evaluation_stats nested)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(agents, fh, indent=2, default=str)
