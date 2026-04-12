"""Tests for the theme system."""

import pytest

from deepbullwhip.render.theme import (
    EdgeStyle,
    FigureStyle,
    FontStyle,
    NodeStyle,
    Theme,
    get_theme,
    list_themes,
    register_theme,
)


class TestBuiltInThemes:
    def test_kfupm_exists(self):
        theme = get_theme("kfupm")
        assert theme.name == "kfupm"
        assert theme.figure.width == 7.0
        assert theme.font.family == "sans-serif"

    def test_ieee_exists(self):
        theme = get_theme("ieee")
        assert theme.name == "ieee"
        assert theme.figure.width == 3.5
        assert theme.font.node_label_size == 7.0

    def test_presentation_exists(self):
        theme = get_theme("presentation")
        assert theme.name == "presentation"
        assert theme.figure.width == 10.0
        assert theme.font.family == "sans-serif"

    def test_minimal_exists(self):
        theme = get_theme("minimal")
        assert theme.name == "minimal"
        assert theme.node.fill_alpha == 0.90

    def test_all_themes_listed(self):
        themes = list_themes()
        assert "kfupm" in themes
        assert "ieee" in themes
        assert "presentation" in themes
        assert "minimal" in themes
        assert len(themes) >= 4


class TestThemeOverride:
    def test_override_font(self):
        theme = get_theme("kfupm")
        big = theme.override(font=FontStyle(node_label_size=14.0))
        assert big.font.node_label_size == 14.0
        assert big.name == "kfupm"  # unchanged
        assert big.figure.width == 7.0  # unchanged

    def test_override_figure(self):
        theme = get_theme("ieee")
        wide = theme.override(figure=FigureStyle(width=10.0))
        assert wide.figure.width == 10.0
        assert wide.font.node_label_size == 7.0  # unchanged

    def test_override_preserves_immutability(self):
        theme = get_theme("kfupm")
        modified = theme.override(name="modified_kfupm")
        assert theme.name == "kfupm"  # original unchanged
        assert modified.name == "modified_kfupm"


class TestThemeColors:
    def test_node_color_cycles(self):
        theme = get_theme("kfupm")
        c0 = theme.node_color(0)
        c6 = theme.node_color(6)
        assert c0 == c6  # cycles after len(colors)

    def test_bw_color_low(self):
        theme = get_theme("kfupm")
        assert theme.bw_color(1.0) == theme.bw_colors["low"]

    def test_bw_color_medium(self):
        theme = get_theme("kfupm")
        assert theme.bw_color(2.0) == theme.bw_colors["medium"]

    def test_bw_color_high(self):
        theme = get_theme("kfupm")
        assert theme.bw_color(4.0) == theme.bw_colors["high"]


class TestThemeRegistry:
    def test_unknown_theme_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_theme("nonexistent_theme")

    def test_register_custom_theme(self):
        custom = Theme(
            name="test_custom",
            colors=["#FF0000"],
            bw_colors={"low": "#00FF00", "medium": "#FFFF00", "high": "#FF0000"},
        )
        register_theme("test_custom", custom)
        retrieved = get_theme("test_custom")
        assert retrieved.colors == ["#FF0000"]
        # Cleanup
        from deepbullwhip.render.theme import _THEMES
        del _THEMES["test_custom"]


class TestStyleDataclasses:
    def test_node_style_defaults(self):
        style = NodeStyle()
        assert style.shape == "circle"
        assert style.fill_alpha == 0.90

    def test_edge_style_defaults(self):
        style = EdgeStyle()
        assert style.arrow_style == "-|>"
        assert style.color == "#888888"

    def test_font_style_defaults(self):
        style = FontStyle()
        assert style.family == "serif"
        assert style.title_weight == "bold"

    def test_figure_style_defaults(self):
        style = FigureStyle()
        assert style.width == 7.0
        assert style.height is None
        assert style.dpi == 300

    def test_frozen_immutability(self):
        style = NodeStyle()
        with pytest.raises(AttributeError):
            style.fill_alpha = 0.5  # type: ignore
