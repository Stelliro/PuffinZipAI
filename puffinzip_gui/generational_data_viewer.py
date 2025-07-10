# PuffinZipAI_Project/puffinzip_gui/generational_data_viewer.py
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import numpy as np
import logging
import sys

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


class GenerationalDataViewerTab(ttk.Frame):
    def __init__(self, parent, app_instance, els_optimizer_instance, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app_instance
        self.logger = getattr(app_instance, 'logger', None)
        if not self.logger:

            self.logger = logging.getLogger("GenDataViewerFallbackLogger_PMA")
            if not self.logger.handlers:
                _fb_handler_gdv = logging.StreamHandler(sys.stdout)
                _fb_formatter_gdv = logging.Formatter('%(asctime)s - GDV_FB_PMA - %(levelname)s - %(message)s')
                _fb_handler_gdv.setFormatter(_fb_formatter_gdv)
                self.logger.addHandler(_fb_handler_gdv)
                self.logger.setLevel(logging.INFO)
            self.logger.warning("Using fallback logger for GenerationalDataViewerTab, configured within PMA structure.")

        self.els_optimizer = els_optimizer_instance
        self.current_selected_generation_data = []

        self.frame_bg = getattr(self.app, 'FRAME_BG', FRAME_BG_FALLBACK)
        self.label_fg = getattr(self.app, 'LABEL_FG', LABEL_FG_FALLBACK)
        self.input_bg = getattr(self.app, 'INPUT_BG', INPUT_BG_FALLBACK)
        self.input_fg = getattr(self.app, 'INPUT_FG', INPUT_FG_FALLBACK)
        self.accent_color = getattr(self.app, 'ACCENT_COLOR', ACCENT_COLOR_FALLBACK)

        self.font_normal = getattr(self.app, 'FONT_NORMAL', FONT_NORMAL_FALLBACK)
        self.font_bold = getattr(self.app, 'FONT_BOLD', FONT_BOLD_FALLBACK)
        self.font_section_title = getattr(self.app, 'FONT_SECTION_TITLE', FONT_SECTION_TITLE_FALLBACK)
        self.font_small = getattr(self.app, 'FONT_SMALL', FONT_SMALL_FALLBACK)

        self.treeview_bg = getattr(self.app, 'TEXT_AREA_BG', TREEVIEW_BG_FALLBACK)
        self.treeview_fg = getattr(self.app, 'TEXT_AREA_FG', TREEVIEW_FG_FALLBACK)
        self.treeview_heading_bg = getattr(self.app, 'BUTTON_BG', TREEVIEW_HEADING_BG_FALLBACK)
        self.treeview_heading_fg = getattr(self.app, 'BUTTON_FG', TREEVIEW_HEADING_FG_FALLBACK)
        self.treeview_selected_bg = getattr(self.app, 'ACCENT_COLOR', TREEVIEW_SELECTED_BG_FALLBACK)
        self.treeview_selected_fg = getattr(self.app, 'TEXT_AREA_BG', TREEVIEW_SELECTED_FG_FALLBACK)

        self.configure(style=f"{STYLE_PREFIX_GDV}TFrame")
        self._setup_internal_styles()
        self._create_widgets()
        self.logger.info("GenerationalDataViewerTab initialized.")

    def _setup_internal_styles(self):
        style = ttk.Style(self)
        style.configure(f"{STYLE_PREFIX_GDV}TFrame", background=self.frame_bg)
        style.configure(f"{STYLE_PREFIX_GDV}TLabel", background=self.frame_bg, foreground=self.label_fg,
                        font=self.font_normal)
        style.configure(f"{STYLE_PREFIX_GDV}Title.TLabel", background=self.frame_bg, foreground=self.accent_color,
                        font=self.font_section_title)
        style.configure(f"{STYLE_PREFIX_GDV}TButton",
                        font=getattr(self.app, 'FONT_BUTTON', FONT_BOLD_FALLBACK))
        style.configure(f"{STYLE_PREFIX_GDV}TEntry",
                        fieldbackground=self.input_bg, foreground=self.input_fg,
                        insertbackground=self.label_fg, font=self.font_normal, padding=4)
        style.map(f"{STYLE_PREFIX_GDV}TEntry", bordercolor=[('focus', self.accent_color)])

        style.configure(f"{STYLE_PREFIX_GDV}Treeview",
                        background=self.treeview_bg,
                        fieldbackground=self.treeview_bg,
                        foreground=self.treeview_fg,
                        font=self.font_small,
                        rowheight=int(self.font_small[1] * 2.2) if self.font_small and len(self.font_small) > 1 else 25)

        style.configure(f"{STYLE_PREFIX_GDV}Treeview.Heading",
                        background=self.treeview_heading_bg,
                        foreground=self.treeview_heading_fg,
                        font=self.font_bold,
                        relief="flat", padding=(6, 6))
        style.map(f"{STYLE_PREFIX_GDV}Treeview.Heading",
                  background=[('active', self.accent_color), ('!active', self.treeview_heading_bg)],
                  relief=[('active', 'groove')])

        style.map(f"{STYLE_PREFIX_GDV}Treeview",
                  background=[('selected', self.treeview_selected_bg)],
                  foreground=[('selected', self.treeview_selected_fg)])

    def _create_widgets(self):
        controls_frame = ttk.Frame(self, style=f"{STYLE_PREFIX_GDV}TFrame", padding=(10, 10))
        controls_frame.pack(fill=tk.X, pady=(5, 10), padx=10)

        generation_label = ttk.Label(controls_frame, text="Viewing ELS Data:", style=f"{STYLE_PREFIX_GDV}TLabel")
        generation_label.pack(side=tk.LEFT, padx=(0, 10))

        self.generation_info_var = tk.StringVar(value="Current/Final ELS Population")
        self.generation_display_entry = ttk.Entry(controls_frame, textvariable=self.generation_info_var,
                                                  style=f"{STYLE_PREFIX_GDV}TEntry", width=30, state='readonly')
        self.generation_display_entry.pack(side=tk.LEFT, padx=(0, 10))

        refresh_button = ttk.Button(controls_frame, text="🔄 Load/Refresh Data",
                                    style=f"{STYLE_PREFIX_GDV}TButton", command=self.load_and_display_data)
        refresh_button.pack(side=tk.LEFT, padx=(5, 0))

        data_frame = ttk.Frame(self, style=f"{STYLE_PREFIX_GDV}TFrame", padding=10)
        data_frame.pack(fill=tk.BOTH, expand=True)

        self.columns = {
            "agent_id": {"text": "Agent ID", "width": 180, "anchor": tk.W},
            "fitness": {"text": "Fitness", "width": 100, "anchor": tk.E},
            "gen_born": {"text": "Gen Born", "width": 70, "anchor": tk.CENTER},
            "params_lr": {"text": "Learn Rate", "width": 100, "anchor": tk.E},
            "params_er": {"text": "Expl. Rate", "width": 100, "anchor": tk.E},
            "params_rle_min": {"text": "RLE MinRun", "width": 90, "anchor": tk.CENTER},
            "params_thresh": {"text": "Thresholds", "width": 200, "anchor": tk.W}
        }
        column_keys = list(self.columns.keys())

        self.tree = ttk.Treeview(data_frame, columns=column_keys, show="headings",
                                 style=f"{STYLE_PREFIX_GDV}Treeview")

        for col_key in column_keys:
            col_data = self.columns[col_key]
            self.tree.heading(col_key, text=col_data["text"], anchor=col_data.get("heading_anchor", tk.CENTER),
                              command=lambda c=col_key: self._sort_treeview(c, False))
            self.tree.column(col_key, width=col_data["width"], anchor=col_data["anchor"], stretch=tk.YES)

        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview, style="Vertical.TScrollbar")
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview, style="Horizontal.TScrollbar")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<Double-1>", self.on_item_double_click)
        self.logger.debug("GenerationalDataViewerTab widgets created.")
        self.load_and_display_data()

    def on_item_double_click(self, event):
        if not hasattr(self, 'tree'): return
        item_iid = self.tree.focus()
        if not item_iid: return
        self.logger.info(f"GDV: Treeview item double-clicked. IID (Agent ID): {item_iid}")

        agent_details = None
        if self.current_selected_generation_data:
            for agent_data in self.current_selected_generation_data:
                if agent_data.get("agent_id") == item_iid:
                    agent_details = agent_data
                    break

        if agent_details:
            details_message_parts = [f"Agent Details (ID: {agent_details.get('agent_id', 'N/A')})"]
            details_message_parts.append("-" * 30)
            details_message_parts.append(f"  Fitness: {agent_details.get('fitness', float('nan')):.4f}")
            details_message_parts.append(f"  Generation Born: {agent_details.get('generation_born', 'N/A')}")
            parent_ids_list = agent_details.get('parent_ids', [])
            parent_ids_str = ', '.join(parent_ids_list) if parent_ids_list else "N/A (Root/Immigrant)"
            details_message_parts.append(f"  Parent IDs: {parent_ids_str}")

            details_message_parts.append("\n  AI Core Hyperparameters:")
            details_message_parts.append(f"    Learning Rate: {agent_details.get('learning_rate', float('nan')):.5f}")
            details_message_parts.append(
                f"    Discount Factor: {agent_details.get('discount_factor', float('nan')):.5f}")
            details_message_parts.append(
                f"    Exploration Rate: {agent_details.get('exploration_rate', float('nan')):.4f}")
            details_message_parts.append(
                f"    Exploration Decay Rate: {agent_details.get('exploration_decay_rate', float('nan')):.5f}")
            details_message_parts.append(
                f"    Min Exploration Rate: {agent_details.get('min_exploration_rate', float('nan')):.5f}")
            details_message_parts.append(
                f"    RLE Min Encodable Run: {agent_details.get('rle_min_encodable_run_length', 'N/A')}")

            thresholds_list = agent_details.get('len_thresholds', [])
            thresholds_str = ", ".join(map(str, thresholds_list)) if thresholds_list else "N/A"
            details_message_parts.append(f"    Length Thresholds: [{thresholds_str}]")

            q_table_shape = agent_details.get('q_table_shape', 'N/A')
            q_table_shape_str = f"{q_table_shape[0]}x{q_table_shape[1]}" if isinstance(q_table_shape, tuple) and len(
                q_table_shape) == 2 else str(q_table_shape)
            details_message_parts.append(f"    Q-Table Shape: {q_table_shape_str}")

            target_device_val = agent_details.get('target_device', 'Unknown')
            details_message_parts.append(f"    Target Device: {target_device_val}")

            eval_stats = agent_details.get('evaluation_stats', {})
            if eval_stats:
                details_message_parts.append("\n  Evaluation Stats (from last eval):")
                stats_items = [
                    ("Total Reward", "total_reward", "{:.4f}"),
                    ("Items Evaluated", "items_evaluated", "{}"),
                    ("Successful RLE Ops", "successful_rle", "{}"),
                    ("RLE Expansions", "rle_expansion", "{}"),
                    ("Chose NoCompression", "chose_nocompression", "{}"),
                    ("Chose AdvancedRLE", "chose_adv_rle", "{}"),
                    ("Decompression Mismatches", "decomp_failures_mismatch", "{}"),
                    ("RLE Errors Returned", "rle_errors_returned", "{}"),
                    ("Avg Proc Time (ms/item)", "total_processing_time_ms", "{:.2f} ms (avg)")
                ]
                items_evald = float(eval_stats.get("items_evaluated", 0))
                for label, key, fmt in stats_items:
                    val = eval_stats.get(key)
                    if val is not None:
                        if key == "total_processing_time_ms" and items_evald > 0:
                            val_display = val / items_evald
                        else:
                            val_display = val
                        details_message_parts.append(f"    {label}: {fmt.format(val_display)}")
            else:
                details_message_parts.append("\n  Evaluation Stats: Not available")

            details_message_show = "\n".join(details_message_parts)
            messagebox.showinfo("Agent Details", details_message_show, parent=self)

            if hasattr(self.app, 'log_message'):
                log_friendly_message = details_message_show.replace("\n", " | ").replace("    ", "")
                self.app.log_message(f"GDV Details Displayed: {log_friendly_message}", "info_no_prefix")
        else:
            self.logger.warning(f"GDV: No detailed data found for agent IID: {item_iid}")
            messagebox.showwarning("Details Not Found",
                                   f"Could not retrieve detailed information for agent ID: {item_iid}", parent=self)

    def _sort_treeview(self, col, reverse):
        if not hasattr(self, 'tree'): return
        try:
            data_list = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        except tk.TclError:
            self.logger.warning("GDV: TclError encountered getting tree data for sort (widget may be gone).")
            return
        except Exception as e_get_data:
            self.logger.error(f"GDV: Unexpected error getting data from treeview for sort: {e_get_data}", exc_info=True)
            return

        def sort_key(item):
            val_str = str(item[0]).strip()
            if val_str.upper() == "N/A":
                return (float('-inf') if not reverse else float('inf'))

            if col in ["fitness", "params_lr", "params_er", "gen_born", "params_rle_min"]:
                try:
                    return float(val_str)
                except ValueError:
                    pass
            return val_str.lower()

        try:
            data_list.sort(key=sort_key, reverse=reverse)
        except Exception as e_sort:
            self.logger.error(f"GDV: Error during data_list.sort: {e_sort}", exc_info=True)
            return

        for index, (val, k) in enumerate(data_list):
            try:
                self.tree.move(k, '', index)
            except tk.TclError:
                self.logger.warning(f"GDV: TclError moving item {k} during sort. Item might be gone.")
                continue
            except Exception as e_move:
                self.logger.error(f"GDV: Unexpected error moving item {k} in treeview: {e_move}", exc_info=True)
                break

        self.tree.heading(col, command=lambda c=col: self._sort_treeview(c, not reverse))
        self.logger.debug(f"GDV: Sorted column '{col}' {'descending' if reverse else 'ascending'}.")

    def load_and_display_data(self):
        self.logger.info("GDV: Attempting to load and display generational data...")
        if hasattr(self, 'tree'):
            for item in self.tree.get_children():
                self.tree.delete(item)
        else:
            self.logger.error("GDV: Treeview widget not found. Cannot clear or load data.")
            if hasattr(self, 'generation_info_var'): self.generation_info_var.set("Error: UI not ready")
            return

        if not self.els_optimizer:
            self.logger.warning("GDV: ELS Optimizer not available. Cannot load data.")
            if hasattr(self.app, 'log_message'):
                self.app.log_message("Generational Deep Dive: Evolutionary Optimizer is not initialized.",
                                     "warning_no_prefix")
            if hasattr(self, 'generation_info_var'): self.generation_info_var.set("ELS Optimizer Not Ready")
            self.tree.insert("", tk.END, values=("ELS Optimizer not available.", "", "", "", "", "", ""),
                             tags=('error_row',))
            return

        population_to_display = []
        run_status_info = "No ELS Data"

        if self.app.task_running and self.app.current_task_is_els:
            if self.els_optimizer.population:
                population_to_display = list(self.els_optimizer.population)
                current_gen_num = self.els_optimizer.total_generations_elapsed + (0 if self.app.els_is_paused else 1)
                run_status_info = f"Live - Gen {current_gen_num} (Paused: {self.app.els_is_paused})"
            else:
                run_status_info = "ELS Running - Population not yet available"
        elif self.app.els_run_completed and self.els_optimizer.population:
            population_to_display = list(self.els_optimizer.population)
            run_status_info = f"Final - Gen {self.els_optimizer.total_generations_elapsed} (Completed)"
        elif self.els_optimizer.population and not self.app.task_running:
            population_to_display = list(self.els_optimizer.population)
            run_status_info = f"Previous Run - Gen {self.els_optimizer.total_generations_elapsed} (Idle)"
        else:
            run_status_info = "No Active or Previously Completed ELS Population"

        if hasattr(self, 'generation_info_var'):
            self.generation_info_var.set(run_status_info + f" - {len(population_to_display)} agents")

        if not population_to_display:
            self.logger.info("GDV: No population data to display for current ELS state.")
            self.tree.insert("", tk.END,
                             values=("No agent data available for the current state.", "", "", "", "", "", ""),
                             tags=('placeholder_row',))
            return

        self.current_selected_generation_data = []
        try:
            sorted_population = sorted(population_to_display, key=lambda agent: agent.get_fitness() if hasattr(agent,
                                                                                                               'get_fitness') else -float(
                'inf'), reverse=True)
        except Exception as e_sort:
            self.logger.error(f"GDV: Error sorting population: {e_sort}", exc_info=True)
            self.tree.insert("", tk.END, values=("Error sorting agent data.", "", "", "", "", "", ""),
                             tags=('error_row',))
            return

        for agent_instance in sorted_population:
            try:
                ai_core = agent_instance.get_puffin_ai()
                if ai_core:
                    thresholds_str = ", ".join(map(str, ai_core.len_thresholds)) if ai_core.len_thresholds else "N/A"
                    agent_id_display = agent_instance.agent_id
                    if len(agent_id_display) > 20 and "..." not in agent_id_display:
                        agent_id_display = "..." + agent_id_display[-15:]

                    fitness_val = agent_instance.get_fitness() if hasattr(agent_instance, 'get_fitness') else float(
                        'nan')

                    values = (
                        agent_id_display,
                        f"{fitness_val:.4f}" if np.isfinite(fitness_val) else "N/A",
                        agent_instance.generation_born,
                        f"{ai_core.learning_rate:.5f}" if hasattr(ai_core, 'learning_rate') else "N/A",
                        f"{ai_core.exploration_rate:.4f}" if hasattr(ai_core, 'exploration_rate') else "N/A",
                        getattr(ai_core, 'rle_min_encodable_run_length', "N/A"),
                        thresholds_str
                    )
                    self.tree.insert("", tk.END, values=values, iid=agent_instance.agent_id)

                    agent_data_for_details = {
                        "agent_id": agent_instance.agent_id,
                        "fitness": fitness_val,
                        "generation_born": agent_instance.generation_born,
                        "parent_ids": list(agent_instance.parent_ids) if hasattr(agent_instance, 'parent_ids') else []
                    }
                    if ai_core:
                        agent_data_for_details.update({
                            "learning_rate": getattr(ai_core, 'learning_rate', float('nan')),
                            "discount_factor": getattr(ai_core, 'discount_factor', float('nan')),
                            "exploration_rate": getattr(ai_core, 'exploration_rate', float('nan')),
                            "exploration_decay_rate": getattr(ai_core, 'exploration_decay_rate', float('nan')),
                            "min_exploration_rate": getattr(ai_core, 'min_exploration_rate', float('nan')),
                            "rle_min_encodable_run_length": getattr(ai_core, 'rle_min_encodable_run_length', 'N/A'),
                            "len_thresholds": list(getattr(ai_core, 'len_thresholds', [])),
                            "q_table_shape": ai_core.q_table.shape if hasattr(ai_core,
                                                                              'q_table') and ai_core.q_table is not None else "N/A",
                            "target_device": getattr(ai_core, 'target_device', 'Unknown'),
                            "evaluation_stats": getattr(agent_instance, 'evaluation_stats', {}).copy()
                        })
                    self.current_selected_generation_data.append(agent_data_for_details)
                else:
                    self.logger.warning(f"Agent {agent_instance.agent_id} has no PuffinZipAI core. Skipping display.")
            except Exception as e_agent_data:
                self.logger.error(
                    f"GDV: Error processing agent {getattr(agent_instance, 'agent_id', 'Unknown')} for display: {e_agent_data}",
                    exc_info=True)
                self.tree.insert("", tk.END,
                                 values=(f"Error: {getattr(agent_instance, 'agent_id', 'Unknown')}", "Error", "", "",
                                         "", "", ""), tags=('error_row',))

        self.logger.info(f"GDV: Displayed {len(self.tree.get_children())} agents.")
        if not self.tree.get_children():
            self.tree.insert("", tk.END, values=("No processable agent data found.", "", "", "", "", "", ""),
                             tags=('placeholder_row',))

        if hasattr(self.tree, 'tag_configure'):
            error_color = getattr(self.app, 'ERROR_FG_COLOR', 'red')
            placeholder_color = getattr(self.app, 'DISABLED_FG_COLOR', 'grey')
            font_for_tag = self.font_small # Use the class's own initialized font
            self.tree.tag_configure('error_row', foreground=error_color, font=font_for_tag)
            self.tree.tag_configure('placeholder_row', foreground=placeholder_color, font=font_for_tag)