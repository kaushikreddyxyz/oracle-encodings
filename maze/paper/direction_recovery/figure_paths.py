"""Locations for final figures and distinct supplementary views."""
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / 'figures'


def figure_path(name, extension):
    if name == 'alignment_violins_all_directions':
        directory = FIGURES / 'supplementary'
    elif name.startswith('alignment_density_g'):
        directory = FIGURES / 'by_maze_size'
    else:
        directory = FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{name}.{extension}'
