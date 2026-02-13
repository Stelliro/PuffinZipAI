"""
PuffinZipAI Web UI Server - Repaired & Robust
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import queue
import json
import os
import sys
import logging
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from puffinzip_ai import PuffinZipAI, EvolutionaryOptimizer
    from puffinzip_ai.config import LOGS_DIR_PATH, APP_VERSION, DEFAULT_NUM_GENERATIONS, DEFAULT_POPULATION_SIZE
    from webui_theme_manager import get_theme_manager
except ImportError as e:
    print(f"ERROR: Could not import PuffinZipAI: {e}")
    sys.exit(1)

# FORCE DEBUG LOGGING to ensure "Fitness:" messages are generated
logging.getLogger('puffinzip_ai').setLevel(logging.DEBUG)

app = Flask(__name__, static_folder='webui_static', template_folder='webui_templates')
CORS(app)
app.logger.setLevel(logging.INFO)

# Global State
class AppState:
    def __init__(self):
        self.log_queue = queue.Queue(maxsize=10000)
        self.is_training = False
        self.training_thread = None
        self.current_generation = 0
        self.current_fitness = 0.0
        self.current_compression_ratio = 0.0
        self.metrics_history = []
        self.training_config = {}
        self.checkpoints = {}
        self.log_file = os.path.join(LOGS_DIR_PATH, 'webui.log')
        self.success_validation = None
        os.makedirs(LOGS_DIR_PATH, exist_ok=True)
        
    def add_log(self, level, message, timestamp=None):
        if timestamp is None: timestamp = datetime.now().isoformat()
        entry = {'timestamp': timestamp, 'level': level, 'message': message}
        try: self.log_queue.put_nowait(entry)
        except: pass
        
app_state = AppState()

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html', version=APP_VERSION)

@app.route('/api/status')
def get_status():
    return jsonify({
        'is_training': app_state.is_training,
        'current_generation': app_state.current_generation,
        'current_fitness': app_state.current_fitness,
        'metrics_count': len(app_state.metrics_history)
    })

@app.route('/api/logs')
def get_logs():
    logs = []
    while not app_state.log_queue.empty():
        try: logs.append(app_state.log_queue.get_nowait())
        except: break
    return jsonify({'logs': logs})

@app.route('/api/metrics')
def get_metrics(): return jsonify({'metrics': app_state.metrics_history})

@app.route('/api/validation')
def get_validation(): return jsonify({'success': True, 'validation': app_state.success_validation})

@app.route('/api/training/start', methods=['POST'])
def start_training():
    if app_state.is_training: return jsonify({'success': False, 'message': 'Running'}), 400
    data = request.get_json() or {}
    
    def worker():
        app_state.is_training = True
        app_state.metrics_history.clear()
        app_state.current_generation = 0
        
        try:
            app_state.add_log('INFO', '🚀 Starting AI Session...')
            
            # Config
            gens = 999999 if data.get('infinite_generations') else int(data.get('generations', 10))
            pop = int(data.get('population_size', 20))
            
            # Stop Event
            real_stop_event = threading.Event()
            def monitor():
                while app_state.is_training: time.sleep(0.5)
                real_stop_event.set()
            threading.Thread(target=monitor, daemon=True).start()

            # --- THE BRIDGE ---
            class WebUIBridgeQueue:
                def put_nowait(self, item):
                    if not isinstance(item, str): return
                    
                    # 1. LIVE GRAPHING (The Fix)
                    if "Fitness:" in item:
                        try:
                            # Extract number after "Fitness:" safely
                            val = float(item.split("Fitness:")[1].strip().split()[0].replace(',', ''))
                            app_state.current_fitness = val
                            app_state.metrics_history.append({
                                'generation': app_state.current_generation,
                                'fitness': val,
                                'compression_ratio': 0.0,
                                'evolution_time': 0.0,
                                'timestamp': datetime.now().isoformat()
                            })
                            # DEBUG PRINT to confirm it's working
                            print(f"[BRIDGE] Plotted: {val}")
                        except: pass

                    # 2. GENERATION TRACKING
                    if "--- ELS Gen" in item:
                        try:
                            gen = int(item.split("Gen")[1].split("/")[0].strip())
                            app_state.current_generation = gen
                        except: pass

                    # 3. LOGGING
                    clean = item.replace('[ELS]', '').strip()
                    level = 'WARNING' if 'WARNING' in clean.upper() else 'ERROR' if 'ERROR' in clean.upper() else 'INFO'
                    app_state.add_log(level, clean)

            # Initialize Optimizer (Safe CPU Mode first to ensure stability)
            optimizer = EvolutionaryOptimizer(
                population_size=pop,
                num_generations=gens,
                gui_output_queue=WebUIBridgeQueue(),
                gui_stop_event=real_stop_event,
                target_device="CPU" 
            )
            optimizer.start_evolution()
            app_state.add_log('INFO', '✅ Training Finished.')
            
        except Exception as e:
            app_state.add_log('ERROR', f'Error: {e}')
            import traceback; traceback.print_exc()
        finally:
            app_state.is_training = False

    app_state.training_thread = threading.Thread(target=worker, daemon=True)
    app_state.training_thread.start()
    return jsonify({'success': True})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    app_state.is_training = False
    return jsonify({'success': True})

@app.route('/api/training/reset', methods=['POST'])
def reset_training():
    app_state.metrics_history = []
    app_state.current_generation = 0
    return jsonify({'success': True})

# Stubs
@app.route('/api/compression-methods')
def stub1(): return jsonify({'methods': [], 'count': 0})
@app.route('/api/themes')
def stub2(): return jsonify({'themes': [], 'count': 0})
@app.route('/api/checkpoints', methods=['GET'])
def stub3(): return jsonify({'checkpoints': [], 'count': 0})

if __name__ == '__main__':
    print("--- WebUI Server Fixed & Ready on 5001 ---")
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
