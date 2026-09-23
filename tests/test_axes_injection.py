import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting_modules import core


def test_panel_grid_infers_figure_and_reshapes_a_vertical_axes_vector():
    fig, axes = plt.subplots(3, 1)
    seen = []

    try:
        panels = core.panel_grid(
            ["signal", "noise", "sn"],
            [None],
            lambda ax, row, col: seen.append((ax, row, col)),
            axes=axes,
            sharex=True,
        )

        assert panels.fig is fig
        assert panels.axes.shape == (3, 1)
        assert [ax for ax, _, _ in seen] == list(axes)
        assert [row for _, row, _ in seen] == ["signal", "noise", "sn"]
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_panel_grid_accepts_integer_grid_sizes_and_descriptive_keywords():
    fig, axes = plt.subplots(2, 3, squeeze=False)
    seen = []

    try:
        panels = core.panel_grid(
            row_keys=2,
            col_keys=3,
            draw_panel=lambda ax, row, col: seen.append((ax, row, col)),
            axes=axes,
        )

        assert panels.axes.shape == (2, 3)
        assert [(row, col) for _, row, col in seen] == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
        ]
    finally:
        plt.close(fig)


def test_panel_grid_keeps_caller_axes_in_its_positional_slot():
    fig, ax = plt.subplots()

    try:
        panels = core.panel_grid([None], [None], lambda *_: None, fig, None, ax)

        assert panels.fig is fig
        assert panels.axes[0, 0] is ax
    finally:
        plt.close(fig)


def test_nested_layout_uses_relative_colorbar_height():
    fig = plt.figure()
    outer = fig.add_gridspec(1, 1)
    layout = core.NestedLayout(
        1, 2, outer[0], has_cbar=True, cbar_frac=0.25, hspace=0
    )

    try:
        gs = layout.make_gridspec(fig)
        panel = fig.add_subplot(gs[0, 0])
        cax = layout.cbar_ax(fig)

        assert cax.get_position().height / panel.get_position().height == pytest.approx(
            0.25
        )
    finally:
        plt.close(fig)


def test_grid_layout_keeps_automatic_colorbars_in_gridspec():
    layout = core.GridLayout(1, 2, has_cbar=True, cbar_height=0.30, cbar_gap=0.20)
    fig = layout.make_figure()

    try:
        panels = layout.make_gridspec(fig)
        panel = fig.add_subplot(panels[0, 0])
        cax = layout.cbar_ax(fig)

        assert cax.get_subplotspec() is not None
        assert cax.get_position().height == pytest.approx(layout.fy(0.30))
        assert panel.get_position().y0 - cax.get_position().y1 == pytest.approx(
            layout.fy(0.20)
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cbar_height": 0}, "cbar_height"),
        ({"cbar_height": -0.1}, "cbar_height"),
        ({"cbar_gap": -0.1}, "cbar_gap"),
    ],
)
def test_grid_layout_rejects_invalid_colorbar_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        core.GridLayout(1, 1, has_cbar=True, **kwargs)
