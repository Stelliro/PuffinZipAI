"""
Cross-Compatible Theme System for PuffinZipAI
Shared between old GUI and web UI
"""

import json
import os
from pathlib import Path

# Theme configuration mapping GUI colors to web UI CSS variables
THEMES_CONFIG = {
    "Nordic Dark (Default)": {
        "THEME_BG_COLOR": "#2E3440",
        "THEME_FG_COLOR": "#ECEFF4",
        "THEME_FRAME_BG": "#3B4252",
        "THEME_ACCENT_COLOR": "#88C0D0",
        "THEME_INPUT_BG": "#434C5E",
        "THEME_TEXT_AREA_BG": "#2E3440",
        "THEME_BUTTON_BG": "#4C566A",
        "THEME_BUTTON_FG": "#ECEFF4",
        "THEME_ERROR_FG": "#BF616A",
        "css_class": "theme-nordic"
    },
    "Dracula": {
        "THEME_BG_COLOR": "#282a36",
        "THEME_FG_COLOR": "#f8f8f2",
        "THEME_FRAME_BG": "#44475a",
        "THEME_ACCENT_COLOR": "#bd93f9",
        "THEME_INPUT_BG": "#44475a",
        "THEME_TEXT_AREA_BG": "#282a36",
        "THEME_BUTTON_BG": "#6272a4",
        "THEME_BUTTON_FG": "#f8f8f2",
        "THEME_ERROR_FG": "#ff5555",
        "css_class": "theme-dracula"
    },
    "Solarized Light": {
        "THEME_BG_COLOR": "#fdf6e3",
        "THEME_FG_COLOR": "#657b83",
        "THEME_FRAME_BG": "#eee8d5",
        "THEME_ACCENT_COLOR": "#268bd2",
        "THEME_INPUT_BG": "#eee8d5",
        "THEME_TEXT_AREA_BG": "#fdf6e3",
        "THEME_BUTTON_BG": "#93a1a1",
        "THEME_BUTTON_FG": "#002b36",
        "THEME_ERROR_FG": "#dc322f",
        "css_class": "theme-solarized-light"
    },
    "Monokai Pro": {
        "THEME_BG_COLOR": "#2D2A2E",
        "THEME_FG_COLOR": "#FCFCFA",
        "THEME_FRAME_BG": "#403E41",
        "THEME_ACCENT_COLOR": "#FFD866",
        "THEME_INPUT_BG": "#403E41",
        "THEME_TEXT_AREA_BG": "#221F22",
        "THEME_BUTTON_BG": "#727072",
        "THEME_BUTTON_FG": "#FCFCFA",
        "THEME_ERROR_FG": "#FF6188",
        "css_class": "theme-monokai"
    },
    "Oceanic Next": {
        "THEME_BG_COLOR": "#1B2B34",
        "THEME_FG_COLOR": "#CDD3DE",
        "THEME_FRAME_BG": "#24343D",
        "THEME_ACCENT_COLOR": "#6699CC",
        "THEME_INPUT_BG": "#2E3F4A",
        "THEME_TEXT_AREA_BG": "#1B2B34",
        "THEME_BUTTON_BG": "#4F6470",
        "THEME_BUTTON_FG": "#CDD3DE",
        "THEME_ERROR_FG": "#EC5f67",
        "css_class": "theme-oceanic"
    },
    "GitHub Dark": {
        "THEME_BG_COLOR": "#0d1117",
        "THEME_FG_COLOR": "#c9d1d9",
        "THEME_FRAME_BG": "#161b22",
        "THEME_ACCENT_COLOR": "#58a6ff",
        "THEME_INPUT_BG": "#0d1117",
        "THEME_TEXT_AREA_BG": "#010409",
        "THEME_BUTTON_BG": "#21262d",
        "THEME_BUTTON_FG": "#c9d1d9",
        "THEME_ERROR_FG": "#f85149",
        "css_class": "theme-github-dark"
    },
    "Zenburn": {
        "THEME_BG_COLOR": "#3F3F3F",
        "THEME_FG_COLOR": "#DCDCCC",
        "THEME_FRAME_BG": "#4F4F4F",
        "THEME_ACCENT_COLOR": "#7F9F7F",
        "THEME_INPUT_BG": "#4F4F4F",
        "THEME_TEXT_AREA_BG": "#3F3F3F",
        "THEME_BUTTON_BG": "#6F6F6F",
        "THEME_BUTTON_FG": "#DCDCCC",
        "THEME_ERROR_FG": "#CC9393",
        "css_class": "theme-zenburn"
    },
    "Material Darker": {
        "THEME_BG_COLOR": "#212121",
        "THEME_FG_COLOR": "#EEFFFF",
        "THEME_FRAME_BG": "#303030",
        "THEME_ACCENT_COLOR": "#82AAFF",
        "THEME_INPUT_BG": "#37474F",
        "THEME_TEXT_AREA_BG": "#212121",
        "THEME_BUTTON_BG": "#546E7A",
        "THEME_BUTTON_FG": "#EEFFFF",
        "THEME_ERROR_FG": "#FF5252",
        "css_class": "theme-material"
    },
    "Gruvbox Dark": {
        "THEME_BG_COLOR": "#282828",
        "THEME_FG_COLOR": "#ebdbb2",
        "THEME_FRAME_BG": "#3c3836",
        "THEME_ACCENT_COLOR": "#fabd2f",
        "THEME_INPUT_BG": "#504945",
        "THEME_TEXT_AREA_BG": "#1d2021",
        "THEME_BUTTON_BG": "#665c54",
        "THEME_BUTTON_FG": "#ebdbb2",
        "THEME_ERROR_FG": "#fb4934",
        "css_class": "theme-gruvbox"
    },
    "Tomorrow Night Blue": {
        "THEME_BG_COLOR": "#002451",
        "THEME_FG_COLOR": "#FFFFFF",
        "THEME_FRAME_BG": "#00346E",
        "THEME_ACCENT_COLOR": "#519ABA",
        "THEME_INPUT_BG": "#003F8A",
        "THEME_TEXT_AREA_BG": "#002451",
        "THEME_BUTTON_BG": "#0053B3",
        "THEME_BUTTON_FG": "#FFFFFF",
        "THEME_ERROR_FG": "#FF7575",
        "css_class": "theme-tomorrow-blue"
    },
    "Forest Green": {
        "THEME_BG_COLOR": "#1E352F",
        "THEME_FG_COLOR": "#C1D7C0",
        "THEME_FRAME_BG": "#2A4B42",
        "THEME_ACCENT_COLOR": "#5FAD56",
        "THEME_INPUT_BG": "#335C4A",
        "THEME_TEXT_AREA_BG": "#1E352F",
        "THEME_BUTTON_BG": "#4B8F5A",
        "THEME_BUTTON_FG": "#E0F0DE",
        "THEME_ERROR_FG": "#EF6C64",
        "css_class": "theme-forest-green"
    },
    "Crimson Night": {
        "THEME_BG_COLOR": "#3B0E17",
        "THEME_FG_COLOR": "#FADBD8",
        "THEME_FRAME_BG": "#5C1723",
        "THEME_ACCENT_COLOR": "#E74C3C",
        "THEME_INPUT_BG": "#7A2E3D",
        "THEME_TEXT_AREA_BG": "#3B0E17",
        "THEME_BUTTON_BG": "#A93226",
        "THEME_BUTTON_FG": "#FDEDEC",
        "THEME_ERROR_FG": "#FF9E9A",
        "css_class": "theme-crimson"
    },
    "Electric Blue": {
        "THEME_BG_COLOR": "#0A1931",
        "THEME_FG_COLOR": "#E6F1FF",
        "THEME_FRAME_BG": "#183A5D",
        "THEME_ACCENT_COLOR": "#00BFFF",
        "THEME_INPUT_BG": "#27496D",
        "THEME_TEXT_AREA_BG": "#0A1931",
        "THEME_BUTTON_BG": "#1385D8",
        "THEME_BUTTON_FG": "#E6F1FF",
        "THEME_ERROR_FG": "#FF6B6B",
        "css_class": "theme-electric-blue"
    },
    "Golden Sand": {
        "THEME_BG_COLOR": "#F4E9D8",
        "THEME_FG_COLOR": "#785E48",
        "THEME_FRAME_BG": "#E9DAC7",
        "THEME_ACCENT_COLOR": "#D4A276",
        "THEME_INPUT_BG": "#DFCAAD",
        "THEME_TEXT_AREA_BG": "#F4E9D8",
        "THEME_BUTTON_BG": "#B9926B",
        "THEME_BUTTON_FG": "#543D2B",
        "THEME_ERROR_FG": "#C0392B",
        "css_class": "theme-golden-sand"
    },
    "Neon Glow": {
        "THEME_BG_COLOR": "#100C08",
        "THEME_FG_COLOR": "#F0F0F0",
        "THEME_FRAME_BG": "#2A2015",
        "THEME_ACCENT_COLOR": "#FF00FF",
        "THEME_INPUT_BG": "#3A3025",
        "THEME_TEXT_AREA_BG": "#100C08",
        "THEME_BUTTON_BG": "#D900D9",
        "THEME_BUTTON_FG": "#100C08",
        "THEME_ERROR_FG": "#00FFFF",
        "css_class": "theme-neon-glow"
    },
    "Matrix Green": {
        "THEME_BG_COLOR": "#020F00",
        "THEME_FG_COLOR": "#39FF14",
        "THEME_FRAME_BG": "#041F00",
        "THEME_ACCENT_COLOR": "#00FF00",
        "THEME_INPUT_BG": "#0A3A00",
        "THEME_TEXT_AREA_BG": "#020F00",
        "THEME_BUTTON_BG": "#0F5100",
        "THEME_BUTTON_FG": "#39FF14",
        "THEME_ERROR_FG": "#90EE90",
        "css_class": "theme-matrix-green"
    },
    "Sunset Orange": {
        "THEME_BG_COLOR": "#2C1E32",
        "THEME_FG_COLOR": "#FDD5B7",
        "THEME_FRAME_BG": "#4A2D53",
        "THEME_ACCENT_COLOR": "#FF8C00",
        "THEME_INPUT_BG": "#6F427C",
        "THEME_TEXT_AREA_BG": "#2C1E32",
        "THEME_BUTTON_BG": "#D96E00",
        "THEME_BUTTON_FG": "#FFF0E1",
        "THEME_ERROR_FG": "#FFB366",
        "css_class": "theme-sunset-orange"
    },
    "Lavender Dream": {
        "THEME_BG_COLOR": "#E6E0F8",
        "THEME_FG_COLOR": "#5D5478",
        "THEME_FRAME_BG": "#D1C4E9",
        "THEME_ACCENT_COLOR": "#9575CD",
        "THEME_INPUT_BG": "#B39DDB",
        "THEME_TEXT_AREA_BG": "#E6E0F8",
        "THEME_BUTTON_BG": "#7E57C2",
        "THEME_BUTTON_FG": "#F8F5FF",
        "THEME_ERROR_FG": "#EF5350",
        "css_class": "theme-lavender"
    },
    "Paper White": {
        "THEME_BG_COLOR": "#FFFFFF",
        "THEME_FG_COLOR": "#212121",
        "THEME_FRAME_BG": "#F5F5F5",
        "THEME_ACCENT_COLOR": "#007BFF",
        "THEME_INPUT_BG": "#EEEEEE",
        "THEME_TEXT_AREA_BG": "#FFFFFF",
        "THEME_BUTTON_BG": "#E0E0E0",
        "THEME_BUTTON_FG": "#212121",
        "THEME_ERROR_FG": "#D32F2F",
        "css_class": "theme-paper-white"
    },
    "Coffee House": {
        "THEME_BG_COLOR": "#3E2723",
        "THEME_FG_COLOR": "#D7CCC8",
        "THEME_FRAME_BG": "#4E342E",
        "THEME_ACCENT_COLOR": "#A1887F",
        "THEME_INPUT_BG": "#5D4037",
        "THEME_TEXT_AREA_BG": "#3E2723",
        "THEME_BUTTON_BG": "#795548",
        "THEME_BUTTON_FG": "#EFEBE9",
        "THEME_ERROR_FG": "#FFAB91",
        "css_class": "theme-coffee"
    }
}


class ThemeManager:
    """Manages cross-compatible themes for GUI and Web UI"""
    
    def __init__(self):
        self.themes = THEMES_CONFIG
        self.default_theme = "Nordic Dark (Default)"
    
    def get_all_themes(self):
        """Get all available themes with metadata"""
        return {
            name: {
                'name': name,
                'css_class': info['css_class'],
                'colors': {
                    'bg': info['THEME_BG_COLOR'],
                    'fg': info['THEME_FG_COLOR'],
                    'accent': info['THEME_ACCENT_COLOR'],
                    'error': info['THEME_ERROR_FG']
                }
            }
            for name, info in self.themes.items()
        }
    
    def get_theme(self, theme_name):
        """Get specific theme configuration"""
        if theme_name not in self.themes:
            theme_name = self.default_theme
        
        return {
            'name': theme_name,
            'config': self.themes[theme_name]
        }
    
    def get_theme_names(self):
        """Get list of all theme names"""
        return list(self.themes.keys())
    
    def get_css_class(self, theme_name):
        """Get CSS class for a theme"""
        if theme_name in self.themes:
            return self.themes[theme_name]['css_class']
        return self.themes[self.default_theme]['css_class']
    
    def validate_theme(self, theme_name):
        """Check if theme exists"""
        return theme_name in self.themes or theme_name == self.default_theme


# Global instance
_theme_manager = None


def get_theme_manager():
    """Get or create the global theme manager"""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager
