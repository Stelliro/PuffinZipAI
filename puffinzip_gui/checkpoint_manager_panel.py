# PuffinZipAI - Checkpoint Management GUI
"""
GUI components for managing and comparing evolution checkpoints.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
from typing import Callable, Optional, Dict, List


class CheckpointManagerPanel(ttk.Frame):
    """Panel for managing checkpoints in the GUI."""
    
    def __init__(self, parent, app_instance, els_optimizer_instance, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app_instance
        self.logger = getattr(app_instance, 'logger', logging.getLogger('CheckpointPanel'))
        self.els_optimizer = els_optimizer_instance
        
        # Theme attributes
        self.frame_bg = getattr(app_instance, 'FRAME_BG', '#333333')
        self.label_fg = getattr(app_instance, 'LABEL_FG', '#D4D4D4')
        self.button_bg = getattr(app_instance, 'BUTTON_BG', '#0078D4')
        self.button_fg = getattr(app_instance, 'BUTTON_FG', '#FFFFFF')
        self.input_bg = getattr(app_instance, 'INPUT_BG', '#252525')
        self.accent_color = getattr(app_instance, 'ACCENT_COLOR', '#0078D4')
        
        self.font_normal = getattr(app_instance, 'FONT_NORMAL', ('Segoe UI', 10))
        self.font_small = getattr(app_instance, 'FONT_SMALL', ('Segoe UI', 9))
        self.font_bold = getattr(app_instance, 'FONT_SECTION_TITLE', ('Segoe UI', 11, 'bold'))
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the checkpoint management UI."""
        # Main container with padding
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === SAVE CHECKPOINT SECTION ===
        save_frame = ttk.LabelFrame(main_frame, text="Save Checkpoint", padding=10)
        save_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Checkpoint name input
        name_frame = ttk.Frame(save_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Checkpoint Name:", font=self.font_normal).pack(side=tk.LEFT, padx=(0, 5))
        self.checkpoint_name_var = tk.StringVar(value="checkpoint")
        name_entry = ttk.Entry(name_frame, textvariable=self.checkpoint_name_var, width=30)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Save button
        self.save_checkpoint_btn = ttk.Button(
            save_frame,
            text="Save Current Checkpoint",
            command=self._on_save_checkpoint
        )
        self.save_checkpoint_btn.pack(fill=tk.X, pady=5)
        
        # === CHECKPOINT LIST SECTION ===
        list_frame = ttk.LabelFrame(main_frame, text="Available Checkpoints", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview with scrollbar
        tree_scroll_frame = ttk.Frame(list_frame)
        tree_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        scrollbar = ttk.Scrollbar(tree_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.checkpoints_tree = ttk.Treeview(
            tree_scroll_frame,
            columns=('Generation', 'Fitness', 'Score', 'Dataset', 'Timestamp'),
            height=8,
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.checkpoints_tree.yview)
        
        # Configure columns
        self.checkpoints_tree.column('#0', width=120, heading='Name')
        self.checkpoints_tree.column('Generation', width=80, heading='Gen')
        self.checkpoints_tree.column('Fitness', width=80, heading='Best Fit')
        self.checkpoints_tree.column('Score', width=60, heading='Score')
        self.checkpoints_tree.column('Dataset', width=100, heading='Dataset (KB)')
        self.checkpoints_tree.column('Timestamp', width=140, heading='Timestamp')
        
        self.checkpoints_tree.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        self.refresh_checkpoints_btn = ttk.Button(
            list_frame,
            text="Refresh List",
            command=self._on_refresh_list
        )
        self.refresh_checkpoints_btn.pack(fill=tk.X, pady=5)
        
        # === ACTIONS SECTION ===
        actions_frame = ttk.LabelFrame(main_frame, text="Checkpoint Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        button_frame = ttk.Frame(actions_frame)
        button_frame.pack(fill=tk.X)
        
        self.load_checkpoint_btn = ttk.Button(
            button_frame,
            text="Load Selected",
            command=self._on_load_checkpoint
        )
        self.load_checkpoint_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.compare_checkpoint_btn = ttk.Button(
            button_frame,
            text="Compare...",
            command=self._on_compare_checkpoints
        )
        self.compare_checkpoint_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.delete_checkpoint_btn = ttk.Button(
            button_frame,
            text="Delete Selected",
            command=self._on_delete_checkpoint
        )
        self.delete_checkpoint_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # === STATS DISPLAY ===
        stats_frame = ttk.LabelFrame(main_frame, text="Selected Checkpoint Info", padding=10)
        stats_frame.pack(fill=tk.X)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=80, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.config(state=tk.DISABLED)  # Read-only
        
        # Initial update
        self._on_refresh_list()
    
    def _on_save_checkpoint(self):
        """Save current evolution state as checkpoint."""
        if not self.els_optimizer or not self.app:
            messagebox.showerror("Error", "Evolution optimizer not available.")
            return
        
        checkpoint_name = self.checkpoint_name_var.get().strip()
        if not checkpoint_name:
            messagebox.showwarning("Warning", "Please enter a checkpoint name.")
            return
        
        try:
            self.save_checkpoint_btn.config(state=tk.DISABLED)
            success = self.els_optimizer.save_checkpoint(checkpoint_name)
            
            if success:
                messagebox.showinfo("Success", f"Checkpoint '{checkpoint_name}' saved successfully!")
                self._on_refresh_list()
                self.checkpoint_name_var.set("checkpoint")
            else:
                messagebox.showerror("Error", f"Failed to save checkpoint '{checkpoint_name}'.")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error saving checkpoint: {e}")
        finally:
            self.save_checkpoint_btn.config(state=tk.NORMAL)
    
    def _on_refresh_list(self):
        """Refresh the checkpoint list."""
        if not self.els_optimizer:
            return
        
        # Clear existing items
        for item in self.checkpoints_tree.get_children():
            self.checkpoints_tree.delete(item)
        
        # Load and display checkpoints
        try:
            checkpoints = self.els_optimizer.get_checkpoints_list()
            for cp in checkpoints:
                dataset_size_kb = cp['dataset_size'] / 1024 if cp['dataset_size'] > 0 else 0
                values = (
                    cp['generation'],
                    f"{cp['best_fitness']:.4f}",
                    f"{cp['score']:.2f}",
                    f"{dataset_size_kb:.1f}",
                    cp['timestamp']
                )
                self.checkpoints_tree.insert('', tk.END, text=cp['name'], values=values)
        except Exception as e:
            self.logger.error(f"Error refreshing checkpoint list: {e}", exc_info=True)
    
    def _on_load_checkpoint(self):
        """Load selected checkpoint."""
        selection = self.checkpoints_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a checkpoint to load.")
            return
        
        if not self.els_optimizer or not self.app:
            messagebox.showerror("Error", "Evolution optimizer not available.")
            return
        
        item = selection[0]
        checkpoint_name = self.checkpoints_tree.item(item)['text']
        
        # Find the full checkpoint key
        checkpoints = self.els_optimizer.get_checkpoints_list()
        checkpoint_key = next((cp['key'] for cp in checkpoints if cp['name'] == checkpoint_name), None)
        
        if not checkpoint_key:
            messagebox.showerror("Error", f"Checkpoint '{checkpoint_name}' not found.")
            return
        
        if not messagebox.askyesno("Confirm", 
                                   f"Load checkpoint '{checkpoint_name}'?\nCurrent state will be replaced."):
            return
        
        try:
            self.load_checkpoint_btn.config(state=tk.DISABLED)
            success = self.els_optimizer.load_checkpoint(checkpoint_key)
            
            if success:
                messagebox.showinfo("Success", f"Checkpoint '{checkpoint_name}' loaded successfully!")
                # Trigger UI update if available
                if hasattr(self.app, '_update_els_button_states'):
                    self.app._update_els_button_states()
            else:
                messagebox.showerror("Error", f"Failed to load checkpoint '{checkpoint_name}'.")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error loading checkpoint: {e}")
        finally:
            self.load_checkpoint_btn.config(state=tk.NORMAL)
    
    def _on_delete_checkpoint(self):
        """Delete selected checkpoint."""
        selection = self.checkpoints_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a checkpoint to delete.")
            return
        
        if not self.els_optimizer:
            messagebox.showerror("Error", "Evolution optimizer not available.")
            return
        
        item = selection[0]
        checkpoint_name = self.checkpoints_tree.item(item)['text']
        
        # Find the full checkpoint key
        checkpoints = self.els_optimizer.get_checkpoints_list()
        checkpoint_key = next((cp['key'] for cp in checkpoints if cp['name'] == checkpoint_name), None)
        
        if not checkpoint_key:
            messagebox.showerror("Error", f"Checkpoint '{checkpoint_name}' not found.")
            return
        
        if not messagebox.askyesno("Confirm", 
                                   f"Delete checkpoint '{checkpoint_name}'?\nThis cannot be undone."):
            return
        
        try:
            self.els_optimizer.checkpoint_manager.delete_checkpoint(checkpoint_key)
            messagebox.showinfo("Success", f"Checkpoint '{checkpoint_name}' deleted!")
            self._on_refresh_list()
        except Exception as e:
            self.logger.error(f"Error deleting checkpoint: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error deleting checkpoint: {e}")
    
    def _on_compare_checkpoints(self):
        """Open comparison dialog for two checkpoints."""
        selection = self.checkpoints_tree.selection()
        if len(selection) < 1:
            messagebox.showwarning("Warning", "Please select at least one checkpoint.")
            return
        
        if not self.els_optimizer:
            messagebox.showerror("Error", "Evolution optimizer not available.")
            return
        
        # Get all checkpoints
        checkpoints = self.els_optimizer.get_checkpoints_list()
        if not checkpoints:
            messagebox.showwarning("Warning", "No checkpoints available to compare.")
            return
        
        # Create comparison dialog
        ComparisonDialog(self.app, self.els_optimizer, checkpoints)


class ComparisonDialog(tk.Toplevel):
    """Dialog for comparing two checkpoints."""
    
    def __init__(self, parent, els_optimizer, checkpoints_list: List[Dict]):
        super().__init__(parent)
        self.title("Compare Checkpoints")
        self.geometry("700x500")
        self.els_optimizer = els_optimizer
        self.checkpoints_list = checkpoints_list
        self.logger = getattr(parent, 'logger', logging.getLogger('ComparisonDialog'))
        
        # Theme
        self.frame_bg = getattr(parent, 'FRAME_BG', '#333333')
        self.label_fg = getattr(parent, 'LABEL_FG', '#D4D4D4')
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the comparison UI."""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Selection section
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selection_frame, text="Select two checkpoints to compare:").pack(anchor=tk.W)
        
        # Checkpoint 1
        cp1_frame = ttk.Frame(selection_frame)
        cp1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cp1_frame, text="Checkpoint 1:").pack(side=tk.LEFT, padx=(0, 5))
        self.cp1_var = tk.StringVar()
        cp1_combo = ttk.Combobox(cp1_frame, textvariable=self.cp1_var, 
                                  values=[cp['name'] for cp in self.checkpoints_list],
                                  state='readonly', width=40)
        cp1_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Checkpoint 2
        cp2_frame = ttk.Frame(selection_frame)
        cp2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cp2_frame, text="Checkpoint 2:").pack(side=tk.LEFT, padx=(0, 5))
        self.cp2_var = tk.StringVar()
        cp2_combo = ttk.Combobox(cp2_frame, textvariable=self.cp2_var,
                                  values=[cp['name'] for cp in self.checkpoints_list],
                                  state='readonly', width=40)
        cp2_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Compare button
        ttk.Button(selection_frame, text="Compare", 
                  command=self._on_compare).pack(fill=tk.X, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Comparison Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=20, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, pady=10)
    
    def _on_compare(self):
        """Perform checkpoint comparison."""
        cp1_name = self.cp1_var.get()
        cp2_name = self.cp2_var.get()
        
        if not cp1_name or not cp2_name:
            messagebox.showwarning("Warning", "Please select both checkpoints.")
            return
        
        if cp1_name == cp2_name:
            messagebox.showwarning("Warning", "Please select two different checkpoints.")
            return
        
        try:
            # Get checkpoint keys
            cp1_key = next((cp['key'] for cp in self.checkpoints_list if cp['name'] == cp1_name), None)
            cp2_key = next((cp['key'] for cp in self.checkpoints_list if cp['name'] == cp2_name), None)
            
            if not cp1_key or not cp2_key:
                messagebox.showerror("Error", "One or both checkpoints not found.")
                return
            
            # Get comparison
            comparison = self.els_optimizer.compare_checkpoints(cp1_key, cp2_key)
            
            if not comparison:
                messagebox.showerror("Error", "Failed to compare checkpoints.")
                return
            
            # Display results
            self._display_comparison(comparison)
            
        except Exception as e:
            self.logger.error(f"Error comparing checkpoints: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error comparing checkpoints: {e}")
    
    def _display_comparison(self, comparison: Dict):
        """Display comparison results in text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        cp1 = comparison['checkpoint1']
        cp2 = comparison['checkpoint2']
        diff = comparison['differences']
        better = comparison['better_checkpoint']
        
        text = f"""
CHECKPOINT COMPARISON
{'='*60}

CHECKPOINT 1: {cp1['name']}
  Generation:      {cp1['generation']}
  Best Fitness:    {cp1['best_fitness']:.6f}
  Avg Fitness:     {cp1.get('avg_fitness', 'N/A')}
  Score:           {cp1['score']:.2f}
  Dataset Size:    {cp1['dataset_size'] / 1024:.1f} KB
  Timestamp:       {cp1['timestamp']}

CHECKPOINT 2: {cp2['name']}
  Generation:      {cp2['generation']}
  Best Fitness:    {cp2['best_fitness']:.6f}
  Avg Fitness:     {cp2.get('avg_fitness', 'N/A')}
  Score:           {cp2['score']:.2f}
  Dataset Size:    {cp2['dataset_size'] / 1024:.1f} KB
  Timestamp:       {cp2['timestamp']}

DIFFERENCES
{'='*60}
  Best Fitness Diff:  {diff['best_fitness_diff']:+.6f}
  Avg Fitness Diff:   {diff['avg_fitness_diff']:+.6f}
  Generation Diff:    {diff['generation_diff']:+d}
  
SCORE COMPARISON
  CP1 Score:          {diff['score_comparison']['score1']:.2f}
  CP2 Score:          {diff['score_comparison']['score2']:.2f}
  Difference:         {diff['score_comparison']['difference']:+.2f}
  Improvement:        {diff['score_comparison']['improvement_percent']:+.1f}%

VERDICT: {better.upper()} is BETTER
  {cp1['name'] if better == 'checkpoint1' else cp2['name']} outperformed the other checkpoint
"""
        
        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)
