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


def test_panel_grid_keeps_axes_in_its_legacy_positional_slot():
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


def test_grid_layout_rejects_an_unreserved_colorbar_height():
    layout = core.GridLayout(1, 1, has_cbar=True)

    with pytest.raises(ValueError, match="cbar_height"):
        layout.cbar_ax(object(), height=0.30)


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
