"""
PuffinZipAI Web UI Server
Modern, responsive web interface for PuffinZipAI compression optimizer
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
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from puffinzip_ai import PuffinZipAI, EvolutionaryOptimizer
    from puffinzip_ai.config import LOGS_DIR_PATH, APP_VERSION, DEFAULT_NUM_GENERATIONS, DEFAULT_POPULATION_SIZE
    from puffinzip_ai.logger import setup_logger
    from webui_theme_manager import get_theme_manager
except ImportError as e:
    print(f"ERROR: Could not import PuffinZipAI: {e}")
    sys.exit(1)

# Flask app setup
app = Flask(__name__, static_folder='webui_static', template_folder='webui_templates')
CORS(app)

# Configure logging
app.logger.setLevel(logging.DEBUG)

# Global state
class AppState:
    def __init__(self):
        self.log_queue = queue.Queue(maxsize=10000)
        self.is_training = False
        self.training_thread = None
        self.current_generation = 0
        self.current_fitness = 0.0
        self.current_compression_ratio = 0.0
        self.metrics_history = []
        self.compression_methods = {}
        self.training_config = {}
        self.checkpoints = {}  # Store checkpoints: {checkpoint_name: checkpoint_data}
        self.log_file = os.path.join(LOGS_DIR_PATH, 'webui.log')
        self.success_validation = None  # Store gen 50 success check results
        self.training_file_size_mb = 0.0  # Store file size being trained on
        os.makedirs(LOGS_DIR_PATH, exist_ok=True)
        
    def add_log(self, level, message, timestamp=None):
        """Add message to log queue"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            # Remove oldest entry if queue is full
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_entry)
            except:
                pass
        
        # Also write to file
        self._write_log_file(log_entry)
    
    def _write_log_file(self, log_entry):
        """Write log entry to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

app_state = AppState()


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html', version=APP_VERSION)


@app.route('/api/status')
def get_status():
    """Get current application status"""
    return jsonify({
        'is_training': app_state.is_training,
        'current_generation': app_state.current_generation,
        'current_fitness': app_state.current_fitness,
        'timestamp': datetime.now().isoformat(),
        'metrics_count': len(app_state.metrics_history)
    })


@app.route('/api/logs')
def get_logs():
    """Get recent logs as JSON array"""
    logs = []
    
    # Collect all available logs from queue
    while not app_state.log_queue.empty():
        try:
            logs.append(app_state.log_queue.get_nowait())
        except queue.Empty:
            break
    
    return jsonify({
        'logs': logs,
        'total_count': len(logs)
    })


@app.route('/api/metrics')
def get_metrics():
    """Get metrics history for graphing"""
    return jsonify({
        'metrics': app_state.metrics_history,
        'count': len(app_state.metrics_history)
    })


@app.route('/api/validation')
def get_validation():
    """Get success validation results from generation 50"""
    if app_state.success_validation is None:
        return jsonify({
            'success': False,
            'message': 'No validation performed yet',
            'validation': None
        })
    
    return jsonify({
        'success': True,
        'message': 'Validation results available',
        'validation': app_state.success_validation
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update training configuration"""
    if request.method == 'GET':
        return jsonify(app_state.training_config)
    
    elif request.method == 'POST':
        data = request.get_json()
        app_state.training_config = data
        app_state.add_log('INFO', f'Configuration updated: {json.dumps(data)}')
        return jsonify({'success': True, 'message': 'Configuration saved'})


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start evolutionary learning"""
    if app_state.is_training:
        return jsonify({'success': False, 'message': 'Training already in progress'}), 400
    
    data = request.get_json() or {}
    
    def training_worker():
        """Background training process"""
        app_state.is_training = True
        app_state.current_generation = 0
        
        try:
            app_state.add_log('INFO', '🚀 Starting evolutionary learning session...')
            
            # Get config
            infinite_generations = data.get('infinite_generations', False)
            generations = data.get('generations', DEFAULT_NUM_GENERATIONS) if not infinite_generations else 999999
            population_size = data.get('population_size', DEFAULT_POPULATION_SIZE)
            stopping_criteria = data.get('stopping_criteria', 'best')  # 'best' or 'success'
            
            mode_str = "infinite" if infinite_generations else f"{generations}"
            criteria_str = f"({stopping_criteria})"
            app_state.add_log('INFO', f'Configuration: {mode_str} generations, {population_size} population size - {criteria_str}')
            
            # Initialize optimizer
            optimizer = EvolutionaryOptimizer()
            app_state.add_log('INFO', 'Optimizer initialized')
            
            best_fitness = -float('inf')
            no_improvement_count = 0
            
            # Simulate training loop
            for gen in range(generations):
                if not app_state.is_training:
                    app_state.add_log('WARNING', 'Training stopped by user')
                    break
                
                app_state.current_generation = gen + 1
                
                # Simulated metrics
                fitness = 0.85 + (gen * 0.01) + (0.05 * (1 - 2 * (gen % 2)))
                compression_ratio = 0.45 - (gen * 0.005)
                evolution_time = 2.5 + (gen * 0.1)
                
                metric = {
                    'generation': gen + 1,
                    'fitness': fitness,
                    'compression_ratio': compression_ratio,
                    'evolution_time': evolution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                app_state.metrics_history.append(metric)
                app_state.current_fitness = fitness
                app_state.current_compression_ratio = compression_ratio
                
                app_state.add_log('INFO', 
                    f'Generation {gen + 1}/{mode_str} - '
                    f'Fitness: {fitness:.4f} | '
                    f'Compression: {compression_ratio:.2%} | '
                    f'Time: {evolution_time:.2f}s')
                
                # Check stopping criteria
                if stopping_criteria == 'success':
                    # Stop on success - e.g., when fitness reaches a high threshold or compression is good enough
                    success_threshold = 0.95  # 95% fitness threshold
                    if fitness >= success_threshold:
                        app_state.add_log('INFO', f'✅ SUCCESS criteria achieved at generation {gen + 1}! Fitness: {fitness:.4f}')
                        break
                elif stopping_criteria == 'best':
                    # Stop on best - track improvements and stop if no improvement for N generations
                    if fitness > best_fitness:
                        best_fitness = fitness
                        no_improvement_count = 0
                        app_state.add_log('INFO', f'📈 New best fitness: {fitness:.4f}')
                    else:
                        no_improvement_count += 1
                        # For demo purposes, not stopping on no improvement, just tracking
                
                # Simulate processing time
                time.sleep(0.5)
            
            app_state.add_log('INFO', '✅ Training session completed successfully')
            
        except Exception as e:
            app_state.add_log('ERROR', f'Training error: {str(e)}')
            import traceback
            app_state.add_log('ERROR', traceback.format_exc())
        
        finally:
            app_state.is_training = False
    
    # Start training in background thread
    app_state.training_thread = threading.Thread(target=training_worker, daemon=True)
    app_state.training_thread.start()
    
    # Prepare response
    infinite_gens = data.get('infinite_generations', False)
    gen_count = data.get('generations', DEFAULT_NUM_GENERATIONS) if not infinite_gens else "Infinite"
    stopping = data.get('stopping_criteria', 'best')
    
    return jsonify({
        'success': True,
        'message': 'Training started',
        'generations': gen_count,
        'infinite_generations': infinite_gens,
        'stopping_criteria': stopping
    })


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop current training"""
    if not app_state.is_training:
        return jsonify({'success': False, 'message': 'No training in progress'}), 400
    
    app_state.is_training = False
    app_state.add_log('WARNING', '⏹️ Training stopped by user')
    
    # Wait for thread to finish (with timeout)
    if app_state.training_thread:
        app_state.training_thread.join(timeout=5.0)
    
    return jsonify({
        'success': True,
        'message': 'Training stopped',
        'generations_completed': app_state.current_generation
    })


@app.route('/api/training/reset', methods=['POST'])
def reset_training():
    """Reset training data"""
    app_state.metrics_history = []
    app_state.current_generation = 0
    app_state.current_fitness = 0.0
    app_state.add_log('INFO', '🔄 Training data reset')
    
    return jsonify({'success': True, 'message': 'Training data reset'})


@app.route('/api/compression-methods')
def get_compression_methods():
    """Get available compression methods"""
    try:
        from puffinzip_ai import get_registry
        registry = get_registry()
        
        methods = []
        for method_name, method in registry.methods.items():
            methods.append({
                'name': method.name,
                'language': method.language.value,
                'is_novelty': method.metadata.get('is_novelty', False),
                'patterns': method.metadata.get('patterns', [])[:3]  # First 3 patterns
            })
        
        return jsonify({
            'methods': methods,
            'count': len(methods)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'methods': [],
            'count': 0
        }), 500


@app.route('/api/checkpoints', methods=['GET', 'POST'])
def manage_checkpoints():
    """Get list of checkpoints or create a new checkpoint"""
    if request.method == 'GET':
        # Return list of checkpoints with metadata
        checkpoints_list = []
        for name, data in app_state.checkpoints.items():
            checkpoints_list.append({
                'name': name,
                'timestamp': data.get('timestamp'),
                'generation': data.get('generation'),
                'fitness': data.get('fitness'),
                'compression_ratio': data.get('compression_ratio'),
                'score': data.get('score'),
                'metric_count': len(data.get('metrics', []))
            })
        
        return jsonify({
            'checkpoints': sorted(checkpoints_list, key=lambda x: x['timestamp'], reverse=True),
            'count': len(checkpoints_list)
        })
    
    elif request.method == 'POST':
        # Save new checkpoint
        data = request.get_json() or {}
        checkpoint_name = data.get('name', f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if checkpoint_name in app_state.checkpoints:
            return jsonify({'success': False, 'message': 'Checkpoint already exists'}), 400
        
        # Calculate score: composite of fitness and compression ratio
        score = (app_state.current_fitness * 0.6) + ((1 - app_state.current_compression_ratio) * 0.4) * 100
        
        checkpoint = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'generation': app_state.current_generation,
            'fitness': app_state.current_fitness,
            'compression_ratio': app_state.current_compression_ratio,
            'score': round(score, 2),
            'metrics': app_state.metrics_history.copy(),
            'config': app_state.training_config.copy(),
            'file_size_mb': app_state.training_file_size_mb
        }
        
        app_state.checkpoints[checkpoint_name] = checkpoint
        app_state.add_log('INFO', f'✅ Checkpoint saved: {checkpoint_name}')
        
        return jsonify({
            'success': True,
            'message': f'Checkpoint saved: {checkpoint_name}',
            'checkpoint': checkpoint_name
        })


@app.route('/api/checkpoints/<checkpoint_name>', methods=['GET', 'DELETE'])
def get_or_delete_checkpoint(checkpoint_name):
    """Get checkpoint details or delete it"""
    if checkpoint_name not in app_state.checkpoints:
        return jsonify({'success': False, 'message': 'Checkpoint not found'}), 404
    
    if request.method == 'GET':
        checkpoint = app_state.checkpoints[checkpoint_name]
        return jsonify({
            'success': True,
            'checkpoint': checkpoint
        })
    
    elif request.method == 'DELETE':
        del app_state.checkpoints[checkpoint_name]
        app_state.add_log('INFO', f'🗑️ Checkpoint deleted: {checkpoint_name}')
        return jsonify({
            'success': True,
            'message': f'Checkpoint deleted: {checkpoint_name}'
        })


@app.route('/api/checkpoints/<checkpoint_name>/load', methods=['POST'])
def load_checkpoint(checkpoint_name):
    """Load a checkpoint's metrics and config"""
    if checkpoint_name not in app_state.checkpoints:
        return jsonify({'success': False, 'message': 'Checkpoint not found'}), 404
    
    checkpoint = app_state.checkpoints[checkpoint_name]
    
    # Restore checkpoint data
    app_state.metrics_history = checkpoint.get('metrics', []).copy()
    app_state.training_config = checkpoint.get('config', {}).copy()
    app_state.current_generation = checkpoint.get('generation', 0)
    app_state.current_fitness = checkpoint.get('fitness', 0.0)
    app_state.current_compression_ratio = checkpoint.get('compression_ratio', 0.0)
    app_state.training_file_size_mb = checkpoint.get('file_size_mb', 0.0)
    
    app_state.add_log('INFO', f'📂 Checkpoint loaded: {checkpoint_name}')
    
    return jsonify({
        'success': True,
        'message': f'Checkpoint loaded: {checkpoint_name}',
        'checkpoint': checkpoint
    })


@app.route('/api/checkpoints/compare', methods=['POST'])
def compare_checkpoints():
    """Compare two or more checkpoints"""
    data = request.get_json() or {}
    checkpoint_names = data.get('checkpoints', [])
    
    if len(checkpoint_names) < 2:
        return jsonify({'success': False, 'message': 'Need at least 2 checkpoints to compare'}), 400
    
    comparison = []
    for name in checkpoint_names:
        if name in app_state.checkpoints:
            cp = app_state.checkpoints[name]
            comparison.append({
                'name': name,
                'generation': cp.get('generation'),
                'fitness': cp.get('fitness'),
                'compression_ratio': cp.get('compression_ratio'),
                'score': cp.get('score'),
                'timestamp': cp.get('timestamp'),
                'file_size_mb': cp.get('file_size_mb')
            })
    
    if not comparison:
        return jsonify({'success': False, 'message': 'No valid checkpoints to compare'}), 400
    
    return jsonify({
        'success': True,
        'comparison': comparison,
        'count': len(comparison)
    })


@app.route('/api/themes')
def get_themes():
    """Get all available themes"""
    try:
        theme_manager = get_theme_manager()
        themes = theme_manager.get_all_themes()
        
        return jsonify({
            'themes': themes,
            'default': 'Nordic Dark (Default)',
            'count': len(themes)
        })
    except Exception as e:
        app_state.add_log('ERROR', f'Error loading themes: {str(e)}')
        return jsonify({
            'error': str(e),
            'themes': {},
            'count': 0
        }), 500


@app.route('/api/theme/<theme_name>')
def get_theme(theme_name):
    """Get specific theme configuration"""
    try:
        theme_manager = get_theme_manager()
        if not theme_manager.validate_theme(theme_name):
            return jsonify({'error': 'Theme not found'}), 404
        
        theme = theme_manager.get_theme(theme_name)
        return jsonify(theme)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    app_state.add_log('ERROR', f'Server error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

def run_webui(host='127.0.0.1', port=5000, debug=False):
    """Start the Web UI server"""
    
    app_state.add_log('INFO', f'Starting PuffinZipAI Web UI on {host}:{port}')
    
    print(f"""
    ╔════════════════════════════════════════════╗
    ║     PuffinZipAI Web UI - Starting...        ║
    ╠════════════════════════════════════════════╣
    ║  Open your browser and navigate to:        ║
    ║  http://{host}:{port}                     ║
    ║                                            ║
    ║  Press Ctrl+C to stop the server           ║
    ╚════════════════════════════════════════════╝
    """)
    
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        app_state.add_log('INFO', 'Web UI server stopped')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PuffinZipAI Web UI')
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--public', action='store_true', help='Bind to 0.0.0.0 (public)')
    
    args = parser.parse_args()
    
    host = '0.0.0.0' if args.public else args.host
    
    run_webui(host=host, port=args.port, debug=args.debug)
