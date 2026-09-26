# Code lifecycle: active_support. Summary checks for X22+X17 and X17 attribution.
# Scope and settings: docs/CODE_LIFECYCLE.md; docs/research/method-inventory.json.
"""Validate recorded summary fields; training provenance needs separate review."""
import math

from .compare import SCENARIOS


def validate_group(components, seen, *, mainline=False):
    """Reject duplicate cells; optionally enforce the active summary contract.

    ``components`` maps control/fire-or-mixed/mild/severe to summary payloads.
    Deliberately do not infer training steps, batch, initialization or dataset
    identity from a summary: the historical format does not record them.
    """
    control = components['control']
    cell = tuple(control[key] for key in ('history', 'seed', 'year'))
    if cell in seen:
        raise ValueError(f'duplicate history/seed/year cell: {cell}')
    seen.add(cell)
    if not mainline:
        return
    history, seed, year = cell
    for key, value, allowed in (
        ('history', history, (1, 5)), ('seed', seed, (0, 1, 2)),
        ('year', year, (2021, 2022, 2023)),
    ):
        if type(value) is not int or value not in allowed:
            raise ValueError(f'mainline {key} must be one of {allowed}')
    architecture = {1: 'res18_unet', 5: 'res18_utae'}[history]
    population = {2021: 3181, 2022: 2856, 2023: 2102}[year]
    roles = {'control': ('control', None), 'fire': ('cosine_erm', None),
             'mixed': ('block_specialist', None),
             'mild': ('block_specialist', .25), 'severe': ('block_specialist', .5)}
    pixel_counts = set()
    for role, metadata in components.items():
        method, fraction = roles[role]
        for key, value in zip(('history', 'seed', 'year'), cell):
            if type(metadata.get(key)) is not int or metadata[key] != value:
                raise ValueError(f'{role}: matched history/seed/year differ')
        if metadata.get('architecture') != architecture:
            raise ValueError(f'{role}: mainline requires explicit architecture={architecture}')
        if metadata.get('method') != method or metadata.get('block_fraction') != fraction:
            raise ValueError(f'{role}: mainline requires method={method}, block_fraction={fraction}')
        for scenario in SCENARIOS:
            metrics = metadata['results'][scenario]
            ap = metrics['avg_precision']
            if type(ap) not in (int, float) or not math.isfinite(ap) or not 0 <= ap <= 1:
                raise ValueError(f'{role}/{scenario}: AP must be finite and within [0, 1]')
            if type(metrics.get('sample_count')) is not int or metrics['sample_count'] != population:
                raise ValueError(f'{role}/{scenario}: expected sample_count={population}')
            pixels = metrics.get('pixel_count')
            if type(pixels) is not int or pixels <= 0:
                raise ValueError(f'{role}/{scenario}: positive pixel_count required')
            pixel_counts.add(pixels)
    if len(pixel_counts) != 1:
        raise ValueError('matched scenarios must have equal pixel_count')


def has_recorded_seeds(group):
    return sorted(group['seeds']) == [0, 1, 2]
