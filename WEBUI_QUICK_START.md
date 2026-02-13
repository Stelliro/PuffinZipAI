# PuffinZipAI Web UI - Quick Reference

## 🚀 Quick Start (60 seconds)

### Windows
1. Double-click: `run_webui_windows.bat`
2. Browser opens automatically to `http://localhost:5000`
3. Done! Start training

### Linux/macOS
1. Run: `./run_webui.sh`
2. Browser opens to `http://localhost:5000`
3. Done! Start training

## 📋 Main Controls

### Start Training
1. Go to **Training** tab
2. Set **Generations** (how many iterations)
3. Set **Population Size** (how many methods to test)
4. Click **▶ Start Training**
5. Watch the graph update in real-time

### Monitor Progress
- **Top Graph**: Shows fitness and compression ratio progress
- **Status Cards**: Show current generation and fitness score
- **Metrics Section**: Shows compression and timing stats
- **Logs Tab**: Watch detailed progress messages

### View Results
- **Compression Methods Tab**: See all discovered methods
- **Logs Tab**: Review complete training history
- **Settings Tab**: Export data as CSV or JSON

## 🎨 Interface Tour

### Navigation Bar (Top)
```
🐧 PuffinZipAI v1.0    [● Ready]
```
- Logo and version
- Green dot = idle, Orange dot = training

### Graph Section (Persistent Across Tabs)
```
┌─────────────────────────────┐
│ 📊 Generation Progress   ↻ ⛶ │
├─────────────────────────────┤
│   [Fitness line graph]      │
│   [Compression ratio line]  │
└─────────────────────────────┘
```
- Shows fitness trending up, compression improving
- Updates every 500ms during training

### Tabs
```
🎯 Training | 📜 Logs | 📦 Methods | ⚙️ Settings
```

## 📊 Training Tab Controls

| Control | Purpose | Default |
|---------|---------|---------|
| **Generations** | How many iterations | 10 |
| **Population** | Methods per generation | 50 |
| **Mutation Rate** | Randomness level | 50% |
| **▶ Start** | Begin training | - |
| **⏹ Stop** | Halt training | - |
| **🔄 Reset** | Clear all data | - |

## 📜 Logs Tab Features

| Feature | How to Use |
|---------|-----------|
| **Filter** | Select level: All/Info/Warning/Error/Debug |
| **Auto-scroll** | Check/uncheck to follow new logs |
| **Clear** | Remove all log entries |
| **Export** | Download logs as JSON file |

## 📦 Methods Tab

Displays all compression methods:
- 📦 Standard methods (built-in)
- ✨ Novel methods (AI-discovered)
- Language: Python, Rust, CUDA, or Hybrid
- Patterns: Techniques used in algorithm

## ⚙️ Settings Tab

### Display
- 🌙 **Dark Mode**: Toggle theme
- 📊 **Show Grid**: Graph grid lines
- 🔄 **Refresh Rate**: Log polling speed

### Export
- **CSV Metrics**: Download fitness/compression history
- **JSON Logs**: Download all log entries

### Info
- Current version
- Server address
- Last update time

## 🔥 Common Workflows

### Workflow 1: Quick Training Run
```
1. Training tab → Set generations to 20
2. Click ▶ Start Training
3. Watch graph update top
4. See fitness improving
5. When done, Logs tab shows results
```

### Workflow 2: Method Exploration
```
1. Let training run 5+ generations
2. Go to 📦 Methods tab
3. See novel compression methods discovered
4. See pattern combinations used
5. Count growing as evolution runs
```

### Workflow 3: Monitoring Deep Run
```
1. Start 100+ generation training
2. Keep Logs tab open
3. Watch for INFO messages about discoveries
4. Switch to top to see graph progress
5. Use Clear button to clean old logs
6. Export when complete
```

### Workflow 4: Comparing Results
```
1. Export metrics (Settings → Export Metrics)
2. Export logs (Logs → Export)
3. Open CSV and JSON in your preferred tool
4. Analyze fitness progression
5. Review timestamps and achievements
```

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+E` | Export metrics |
| `Ctrl+Shift+L` | Export logs |
| `F12` | Open browser dev tools |
| `Ctrl+R` | Refresh page |
| `Click Tab` | Switch sections |

## 🎯 Tips & Tricks

### Maximize Performance
- Set **Generations** higher (50+) for deeper exploration
- Set **Population Size** higher (100+) for diversity
- Use **50% Mutation Rate** for balanced evolution

### Improve Compression
- Run for many generations (patience!)
- Higher population = better methods found
- Check Logs for discovery messages
- Watch graph for fitness plateaus

### Monitor Better
- Use **Filter** to see only important events
- Check **↻ Refresh** button on graph if it looks stuck
- Turn off **Auto-scroll** when looking at old logs
- Increase **Refresh Interval** if CPU-constrained

### Export Data
- Always export before closing browser
- CSV is great for Excel/spreadsheets
- JSON preserves all metadata
- Use timestamps to identify sessions

## 🐛 Troubleshooting

### "Server not responding"
- Is the terminal running? Should see Flask output
- Try refreshing page (Ctrl+R)
- Check http://localhost:5000

### "Graph not updating"
- Click ↻ refresh button
- Check page has focus
- Increase Refresh Rate in Settings

### "No logs appearing"
- Logs appear after first generation completes
- May take 2-3 seconds on slow systems
- Check browser console (F12) for errors

### "Lost my data"
- Data is on server, not in browser
- Check logs/ folder for webui.log file
- Try running training again

## 📚 More Information

- **Full Docs**: [WEBUI_DOCUMENTATION.md](WEBUI_DOCUMENTATION.md)
- **Migration Guide**: [WEBUI_MIGRATION_GUIDE.md](WEBUI_MIGRATION_GUIDE.md)
- **Old GUI Still Available**: Run `python run_gui.py`

## 🎮 Let's Get Started!

```bash
# Windows
run_webui_windows.bat

# Linux/macOS
./run_webui.sh

# Manual
python webui_server.py

# Then open browser:
http://localhost:5000
```

**That's it! You're ready to evolve compression algorithms! 🚀**

---

**Need help?** Check the full documentation or review server logs.
