#!/usr/bin/env python3
"""Qualify both real GeoTIFF-to-HDF5 paths on a tiny synthetic fixture in Slurm."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('conversion qualification requires a CPU Slurm allocation')
    for key in ('WILDFIRE_ROOT', 'WILDFIRE_UPSTREAM'):
        if not os.environ.get(key):
            raise ValueError(f'explicit {key} is required')
    # Native libraries and fixture I/O are deliberately below the Slurm guard.
    import h5py
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    root = Path(os.environ['WILDFIRE_ROOT']).expanduser().resolve()
    upstream = Path(os.environ['WILDFIRE_UPSTREAM']).expanduser().resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    qualification = root / 'qualification'
    qualification.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='conversion-', dir=qualification))
    (output / 'QUALIFICATION_ONLY.json').write_text(json.dumps({
        'scientific_claim': False, 'synthetic_fixture': True,
        'slurm_job_id': os.environ['SLURM_JOB_ID']}) + '\n')
    print(f'CONVERSION_QUALIFICATION_ROOT={output}', flush=True)
    raw = output / 'tiffs'
    transform = from_origin(-123.0, 49.0, .01, .01)
    frames = []
    for day in range(3):
        frame = np.arange(23 * 8 * 10, dtype=np.float32).reshape(23, 8, 10) + day * 2000
        frame[22] = 130 + day * 100
        frame[22, 0, 0] = np.nan
        frame[22, 0, 1] = 0
        frame[22, 0, 2] = 2359
        frame[3, 1, 1] = np.nan
        frames.append(frame)
    frames = np.stack(frames)

    def make_event(year):
        event = raw / str(year) / 'qualification_event'
        event.mkdir(parents=True)
        dates = [f'{year}-07-{day:02d}' for day in (1, 2, 3)]
        for date, frame in zip(dates, frames):
            with rasterio.open(event / (date + '.tif'), 'w', driver='GTiff',
                               width=10, height=8, count=23, dtype='float32',
                               crs='EPSG:4326', transform=transform) as handle:
                handle.write(frame)
        return event, dates

    dates_by_year = {}
    for year in (2018, 2019, 2020, 2021):
        _, dates_by_year[year] = make_event(year)
    upstream_output = output / 'upstream-hdf5'
    command = [sys.executable, str(upstream / 'src/preprocess/CreateHDF5Dataset.py'),
               '--data_dir', str(raw), '--target_dir', str(upstream_output)]
    (output / 'upstream-command.json').write_text(json.dumps(command, indent=2) + '\n')
    environment = dict(os.environ, HDF5_USE_FILE_LOCKING='FALSE')
    with (output / 'upstream-conversion.log').open('w') as log:
        subprocess.run(command, cwd=upstream, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)

    def inspect(path, year, dates, *, original):
        expected = frames.copy()
        expected[:, 22] = np.nan_to_num(expected[:, 22], nan=0)
        if original:
            expected[:, 22] = np.floor_divide(expected[:, 22], 100)
        with h5py.File(path, 'r') as handle:
            assert set(handle) == {'data'}, str(path)
            data = handle['data']
            assert data.shape == (3, 23, 8, 10), data.shape
            assert data.dtype == np.dtype('float32'), data.dtype
            np.testing.assert_allclose(data[:], expected, rtol=0, atol=0, equal_nan=True)
            assert int(data.attrs['year']) == year
            assert data.attrs['fire_name'] == 'qualification_event'
            actual_dates = [value.decode() if isinstance(value, bytes) else str(value)
                            for value in data.attrs['img_dates']]
            assert actual_dates == dates, actual_dates
            np.testing.assert_allclose(data.attrs['lnglat'], [-122.95, 48.96], rtol=0, atol=1e-7)
            if not original:
                assert data.compression == 'lzf' and data.shuffle
            return {'file': str(path), 'shape': list(data.shape), 'dtype': str(data.dtype),
                    'dates': actual_dates, 'lnglat': list(data.attrs['lnglat']),
                    'active_fire_conversion': 'NaN-to-zero + floor(HHMM/100)' if original else 'NaN-to-zero; original values retained'}

    results = {'upstream': [], 'added_year': []}
    for year in dates_by_year:
        results['upstream'].append(inspect(upstream_output / str(year) / 'qualification_event.hdf5',
                                           year, dates_by_year[year], original=True))
    # Execute the committed converter itself, avoiding its full-population CLI
    # count gate because this intentionally contains only one synthetic event.
    converter_path = repository / 'docs/tutorials/res18/convert-wstsplus-added.py'
    spec = importlib.util.spec_from_file_location('qualification_added_converter', converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    for year in (2016, 2017, 2022, 2023):
        event, dates = make_event(year)
        target = output / 'added-hdf5' / str(year) / 'qualification_event.hdf5'
        target.parent.mkdir(parents=True)
        converter.convert_event(event, target, year)
        results['added_year'].append(inspect(target, year, dates, original=False))
    report = {'status': 'qualification-pass', 'scientific_claim': False,
              'synthetic_fixture': True, 'full_public_download_qualified': False,
              'full_999_event_population_qualified': False,
              'repair_stage_qualified': False,
              'python': sys.version, 'rasterio': rasterio.__version__,
              'gdal': rasterio.__gdal_version__, 'h5py': h5py.__version__,
              'numpy': np.__version__, 'results': results}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
