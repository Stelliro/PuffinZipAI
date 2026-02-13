# PuffinZipAI Web UI Documentation

## Overview

The PuffinZipAI Web UI is a modern, responsive web-based interface for the evolutionary learning system. It replaces the traditional desktop GUI with a clean, intuitive interface that can be accessed from any device with a web browser.

## Features

### 🎯 Real-Time Training Control
- **Start/Stop/Reset** training sessions with intuitive controls
- **Live Generation Tracking** showing current generation and fitness scores
- **Configurable Parameters** including generations, population size, and mutation rates

### 📊 Persistent Graph Display
- **Generation Progress Graph** that follows across all tabs
- **Dual-Axis Chart** showing both fitness scores and compression ratios
- **Auto-Refresh** capability with manual refresh option
- **Responsive Design** that adapts to any screen size

### 📜 Real-Time Logging
- **Live Log Stream** showing all generation events in real-time
- **Filterable Logging** by level (Info, Warning, Error, Debug)
- **Auto-Scroll** with toggle option
- **Export Logs** as JSON for analysis
- **Persistent Display** of 500 most recent entries

### 📦 Compression Methods Gallery
- **Visual Method Display** with language and type indicators
- **Novel Method Badges** for AI-discovered compression algorithms
- **Pattern Information** showing which techniques are used
- **Real-Time Updates** as new methods are discovered

### ⚙️ Settings & Information
- **Display Preferences** including dark mode toggle
- **Refresh Interval Configuration** for log polling
- **System Information** display
- **Data Export** capabilities for metrics and logs

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Flask and Flask-CORS (automatically installed)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Navigate to the PuffinZipAI directory:**
   ```bash
   cd path/to/PuffinZipAI
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors
   ```

## Running the Web UI

### Windows
Double-click `run_webui_windows.bat` or run:
```bash
run_webui_windows.bat
```

### Linux/macOS
Run the shell script:
```bash
chmod +x run_webui.sh
./run_webui.sh
```

### Manual Start
```bash
python webui_server.py --host 127.0.0.1 --port 5000
```

### Advanced Options
```bash
python webui_server.py --help

Options:
  --host HOST          Server host (default: 127.0.0.1)
  --port PORT          Server port (default: 5000)
  --debug              Enable debug mode
  --public             Bind to 0.0.0.0 (public network access)
```

### Access the Web UI
Once the server is running, open your browser and navigate to:
```
http://localhost:5000
```

## Interface Guide

### Navigation Bar
- **PuffinZipAI Logo** with version information
- **Status Indicator** showing training state (green = idle, orange = training)
- **Current Status** text

### Training Tab (Default)
Controls for running the evolutionary learning system:

**Configuration Section:**
- **Generations**: Number of generations to run (1-1000)
- **Population Size**: Number of individuals per generation (10-500)
- **Mutation Rate**: Probability of mutation (0-100%)

**Control Buttons:**
- **▶ Start Training**: Begin a new training session
- **⏹ Stop Training**: Halt current training
- **🔄 Reset Data**: Clear metrics history

**Status Cards:**
- **Generation**: Current generation number
- **Fitness**: Highest fitness value achieved
- **Status**: Training state (Idle/Training)

**Real-Time Metrics:**
- **Compression Ratio**: Current best compression achieved
- **Evolution Time**: Average time per generation
- **Method Count**: Total compression methods in registry

### Logs Tab
Real-time stream of system events and generation logs:

**Log Controls:**
- **Filter Dropdown**: Filter by log level (All, Info, Warning, Error, Debug)
- **Clear Button**: Remove all log entries
- **Auto-Scroll Toggle**: Auto-scroll to newest logs

**Log Display:**
- Color-coded by level
- Shows timestamp, level, and message
- Scrollable view of last 500 entries

### Compression Methods Tab
Gallery of available compression methods:

**Method Display:**
- **Name** with icon (📦 standard, ✨ novel)
- **Language** where implemented (Python, Rust, CUDA, Hybrid)
- **Badges** indicating special properties
- **Patterns** used in implementation

**Statistics:**
- **Total Methods**: Count of all available methods
- **Novel Methods**: Count of AI-discovered methods

### Settings Tab

**Display Settings:**
- **Show Grid on Graphs**: Toggle grid display
- **Dark Mode**: Toggle between dark/light themes
- **Log Refresh Interval**: Set polling frequency (100-10000 ms)

**System Information:**
- **App Version**: Current PuffinZipAI version
- **Server Host**: Current server address
- **Last Updated**: Last system update timestamp

**Data Export:**
- **Export Logs**: Download all logs as JSON
- **Export Metrics**: Download metrics data as CSV

## Graph Controls

### Persistent Graph Section
The graph at the top of every tab shows:
- **Fitness Score** (blue line) - Main fitness metric
- **Compression Ratio** (purple line) - Compression effectiveness

**Controls:**
- **↻ Refresh Button**: Manually update graph data
- **⛶ Fullscreen Button**: Expand graph to fullscreen

**Features:**
- **Dual-Axis Display**: Compare fitness and compression simultaneously
- **Auto-Update**: Updates every 500ms during training
- **Point Interactivity**: Hover over points to see details
- **Data Limit**: Keeps last 100 data points for performance

## API Reference

The web UI communicates with the backend via REST API:

### Status Endpoints
```
GET /api/status
  Returns: {is_training, current_generation, current_fitness, timestamp, metrics_count}

GET /api/logs
  Returns: {logs: [{timestamp, level, message}], total_count}

GET /api/metrics
  Returns: {metrics: [{generation, fitness, compression_ratio, evolution_time}], count}
```

### Training Control
```
POST /api/training/start
  Body: {generations, population_size}
  Returns: {success, message, generations}

POST /api/training/stop
  Returns: {success, message, generations_completed}

POST /api/training/reset
  Returns: {success, message}
```

### Configuration
```
GET /api/config
  Returns: Current training configuration

POST /api/config
  Body: {key: value, ...}
  Returns: {success, message}
```

### Compression Methods
```
GET /api/compression-methods
  Returns: {methods: [{name, language, is_novelty, patterns}], count}
```

## Configuration

The web UI automatically loads and saves settings:

**Local Storage:**
- `darkMode`: Boolean for light/dark theme
- `refreshInterval`: Log polling interval in milliseconds

**Server-Side Logs:**
- Stored in `logs/webui.log`
- Contains all system and training logs

## Keyboard Shortcuts

- **Tab**: Switch between tabs
- **Ctrl+L**: Focus log filter
- **Ctrl+Shift+E**: Export metrics
- **Ctrl+Shift+L**: Export logs

## Troubleshooting

### Server Won't Start
**Problem:** "Port 5000 already in use"
**Solution:** 
```bash
python webui_server.py --port 5001
```

### Browser Can't Connect
**Problem:** "Cannot reach localhost:5000"
**Solution:**
- Check if server is running
- Try refreshing the page
- Check firewall settings
- Try a different browser

### Logs Not Updating
**Problem:** No new logs appearing
**Solution:**
- Check refresh interval setting (should be 1000ms or less)
- Click the "Clear" button to reset
- Check server logs for errors

### Graph Not Showing Data
**Problem:** Empty graph despite training
**Solution:**
- Click the "Refresh" button on the graph
- Wait for first metric to be recorded
- Check browser console for errors (F12)

### Loss of Data on Refresh
**Problem:** Graphs/logs cleared when page reloads
**Solution:** This is normal - data is stored on the server. Check logs folder or re-run the same training session.

## Performance Optimization

### For Large Datasets
- Reduce **log refresh interval** in settings
- Use **log filter** to focus on relevant messages
- Export and archive old logs to reduce display burden

### For Slower Networks
- Increase **log refresh interval** to 2000-5000ms
- Reduce **graph update frequency** (in app.js: updateMetrics interval)
- Use **fullscreen mode** for graphs to prevent redundant renders

### Browser Memory
- Clear logs periodically using the Clear button
- Close unused tabs
- The app automatically keeps only last 500 log entries

## Advanced Usage

### Remote Access
To access the UI from another machine:

```bash
python webui_server.py --host 0.0.0.0 --port 5000 --public
```

Then access from another machine:
```
http://[server-ip]:5000
```

### Custom Styling
Edit `webui_static/css/style.css` to customize colors and layout.

### API Integration
The REST API can be used to integrate PuffinZipAI with other systems:

```python
import requests

# Get current status
response = requests.get('http://localhost:5000/api/status')
print(response.json())

# Start training
response = requests.post('http://localhost:5000/api/training/start', 
                       json={'generations': 20, 'population_size': 100})
print(response.json())
```

## Comparison: Old GUI vs Web UI

| Feature | Old GUI | Web UI |
|---------|---------|--------|
| **Platform** | Windows Desktop | Browser-based |
| **Performance** | Standard | Optimized |
| **Accessibility** | Local only | Network capable |
| **Modern Design** | Traditional | Contemporary |
| **Real-time Logs** | Limited | Full Stream |
| **Persistent Graphs** | Per-tab | Across Tabs |
| **Mobile Support** | No | Responsive |
| **Customization** | GUI Theme | CSS Customizable |
| **Data Export** | Manual | Automated |

## Migration Guide

The old GUI (`primary_main_app.py`, `secondary_main_app.py`) is still available and functional. You can run both simultaneously or switch between them.

### Using Old GUI
```bash
python run_gui.py
```

### Using New Web UI
```bash
./run_webui_windows.bat  # Windows
./run_webui.sh           # Linux/macOS
```

Both systems:
- Access the same training data
- Use the same configuration files
- Can be run independently or together

## Future Enhancements

Planned features for future releases:
- [ ] WebSocket support for real-time updates (instead of polling)
- [ ] Training history persistence and loading
- [ ] Custom metric tracking and visualization
- [ ] Multi-user support with authentication
- [ ] Job scheduling and automation
- [ ] Advanced data analysis and statistics
- [ ] Integration with external visualization tools
- [ ] Mobile app companion

## Support & Feedback

For issues or suggestions related to the Web UI:
1. Check the logs tab for error messages
2. Review the troubleshooting section
3. Check server console output
4. Review browser console (F12) for JavaScript errors

## License

Same as PuffinZipAI main project.

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Browser Support:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
