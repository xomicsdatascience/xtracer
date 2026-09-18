"""Convert one PAMAF .mbi file to a single-window Bruker TDF .d dataset."""
import argparse
import hashlib
import shutil
import sqlite3
import sys
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
import pyzstd

from xtracer.log import Logger

PROFILE = 'pamaf-single-window-v1'
NUM_SCANS, ALIGNMENT, TIMS_MAX_TOF = 918, 4096, 401563
SCAN_MS, MZ_DELAY, MZ_C0, MZ_TIMEBASE, MZ_C1 = 917.0 / 400.0, 25131.0, 315.70325869866065, 0.2, 154272.1271422364
SQRT_C1_1E6 = np.sqrt(MZ_C1) / 1e6
A0, SLOPE = (MZ_DELAY - MZ_C0) * SQRT_C1_1E6, MZ_TIMEBASE * SQRT_C1_1E6
LEVEL_MSMSTYPE = {1: 9, 2: 0}
TIMS_CALIBRATION = (1.0, 917.0, 226.031615206, 59.7363722417, 27.7995150765, 1.0, 0.0, 129.918158566, 13.168533905599823, 2171.039507429403)



def _template_path():
    return Path(files('xtracer').joinpath('assets', 'bruker_tdf_template', 'analysis.tdf'))


def _encode_type2(counts, tofs, intensities):
    """Encode one TDF compression-type-2 record without per-peak Python loops."""
    peak_count = int(counts.sum())
    values = np.zeros(NUM_SCANS + 2 * peak_count, dtype=np.uint32)
    values[0] = NUM_SCANS
    values[1:NUM_SCANS] = counts[:-1] * 2
    if peak_count:
        deltas = tofs.astype(np.uint32, copy=True)
        deltas += 1
        starts = np.cumsum(counts, dtype=np.int64) - counts
        non_starts = np.ones(peak_count, dtype=bool)
        non_starts[starts[starts < peak_count]] = False
        previous_plus_one = tofs[:-1] + 1
        deltas[1:][non_starts[1:]] -= previous_plus_one[non_starts[1:]]
        locations = NUM_SCANS + 2 * np.arange(peak_count, dtype=np.int64)
        values[locations] = deltas
        values[locations + 1] = intensities
    byte_plane = values.view(np.uint8).reshape(-1, 4).T.reshape(-1)
    compressed = pyzstd.compress(byte_plane.tobytes(), 1)
    return (8 + len(compressed)).to_bytes(4, 'little') + NUM_SCANS.to_bytes(4, 'little') + compressed

def _map_frame(arrival_times, mzs, intensities):
    tofs = np.rint((np.sqrt(mzs.astype(np.float64)) - A0) / SLOPE).astype(np.int64)
    scans = np.rint(917.0 - SCAN_MS * arrival_times.astype(np.float64)).astype(np.int64)
    keep = (tofs >= 0) & (tofs <= TIMS_MAX_TOF) & (scans >= 0) & (scans < NUM_SCANS)
    scans, tofs, intensities = scans[keep], tofs[keep], intensities[keep]
    order = np.lexsort((tofs, scans))
    scans, tofs, intensities = scans[order], tofs[order].astype(np.uint32), intensities[order]
    if tofs.size:
        packed = scans.astype(np.int64) * (TIMS_MAX_TOF + 1) + tofs.astype(np.int64)
        starts = np.flatnonzero(np.r_[True, packed[1:] != packed[:-1]])
        scans, tofs, intensities = scans[starts], tofs[starts], np.add.reduceat(intensities, starts)
    intensities = np.rint(intensities).clip(0, np.iinfo(np.uint32).max).astype(np.uint32)
    return tofs, intensities, np.bincount(scans, minlength=NUM_SCANS).astype(np.uint32)


def _prepare_tdf(tdf_path, input_name, output_frame_count):
    db = sqlite3.connect(tdf_path)
    db.row_factory = sqlite3.Row
    templates = {kind: db.execute('SELECT * FROM Frames WHERE Id=?', (frame,)).fetchone() for kind, frame in [('ms1', 1), ('ms2', 2)]}
    if not all(templates.values()):
        db.close()
        raise RuntimeError('Bundled Bruker TDF template is missing frame templates.')
    props = {kind: db.execute('SELECT Property, Value FROM FrameProperties WHERE Frame=?', (frame,)).fetchall() for kind, frame in [('ms1', 1), ('ms2', 2)]}
    template_property_frames = db.execute('SELECT MAX(Frame) FROM FrameProperties').fetchone()[0] or 0
    columns = [row[1] for row in db.execute('PRAGMA table_info(Frames)')]
    for table in ('DiaFrameMsMsInfo', 'DiaFrameMsMsWindows', 'DiaFrameMsMsWindowGroups', 'FrameMsMsInfo', 'PrmFrameMeasurementMode', 'PrmFrameMsMsInfo', 'ErrorLog', 'Frames'):
        db.execute(f'DELETE FROM {table}')
    db.execute('DELETE FROM FrameProperties WHERE Frame > ?', (output_frame_count,))
    db.execute('UPDATE Segments SET FirstFrame=1, LastFrame=1')
    db.execute('UPDATE MzCalibration SET T1=0,T2=0,dC1=0,dC2=0,C2=0,C3=0,C4=0')
    db.execute('UPDATE TimsCalibration SET ' + ','.join(f'C{i}=?' for i in range(10)) + ' WHERE Id=1', TIMS_CALIBRATION)
    db.executemany('UPDATE GlobalMetadata SET Value=? WHERE Key=?', [('0.5', 'OneOverK0AcqRangeLower'), ('1.78', 'OneOverK0AcqRangeUpper')])
    return db, columns, {k: dict(v) for k, v in templates.items()}, props, template_property_frames


def _insert_frame(db, columns, template, frame_id, rt, msms_type, tims_id, peak_count, max_intensity, summed_intensities):
    row = dict(template)
    row.update(Id=frame_id, Time=float(rt), MsMsType=msms_type, TimsId=int(tims_id), NumPeaks=int(peak_count), MaxIntensity=int(max_intensity), SummedIntensities=int(summed_intensities), NumScans=NUM_SCANS)
    db.execute(f'INSERT INTO Frames ({",".join(columns)}) VALUES ({",".join("?" for _ in columns)})', tuple(row[column] for column in columns))


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{(seconds % 3600) // 60:02}:{seconds % 60:02}'


def _convert(input_mbi, output, logger):
    from xtracer.mbi import MBIReader
    output.mkdir(parents=True)
    shutil.copyfile(_template_path(), output / 'analysis.tdf')
    reader = MBIReader(input_mbi, 2)
    levels = np.asarray(reader.GetFrameMSLevels(), dtype=np.int64)
    rts = np.asarray(reader.GetRetentionTimes(), dtype=np.float64)
    if levels.size != rts.size or levels.size < 2:
        raise RuntimeError('Invalid MBI frame metadata.')
    levels, rts = levels[1:], rts[1:]
    unsupported = sorted(set(levels) - set(LEVEL_MSMSTYPE))
    if unsupported:
        raise RuntimeError(f'Unsupported MBI MS levels: {unsupported}')
    db, columns, templates, props, template_property_frames = _prepare_tdf(
        output / 'analysis.tdf', input_mbi.name, len(levels))
    ms1 = ms2 = total_peaks = 0
    read_seconds = map_seconds = encode_seconds = write_seconds = 0.0
    started = time.perf_counter()
    try:
        with (output / 'analysis.tdf_bin').open('wb') as binary:
            offset = 0
            for output_id, (level, rt) in enumerate(zip(levels, rts), start=1):
                stamp = time.perf_counter()
                at, mz, intensity = reader.get_frame_data(output_id)
                read_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                tofs, encoded, counts = _map_frame(at, mz, intensity)
                map_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                record = _encode_type2(counts, tofs, encoded)
                encode_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                binary.seek(offset)
                binary.write(record)
                msms_type = LEVEL_MSMSTYPE[int(level)]
                kind = 'ms2' if msms_type == 9 else 'ms1'
                _insert_frame(db, columns, templates[kind], output_id, rt, msms_type, offset,
                              len(tofs), int(encoded.max()) if encoded.size else 0,
                              int(encoded.sum(dtype=np.uint64)))
                if output_id > template_property_frames:
                    db.executemany('INSERT INTO FrameProperties(Frame, Property, Value) VALUES (?,?,?)',
                                   [(output_id, item['Property'], item['Value']) for item in props[kind]])
                if msms_type == 9:
                    db.execute('INSERT INTO DiaFrameMsMsInfo(Frame, WindowGroup) VALUES (?,1)', (output_id,))
                    ms2 += 1
                else:
                    ms1 += 1
                total_peaks += len(tofs)
                write_seconds += time.perf_counter() - stamp
                if output_id % 10 == 0 or output_id == len(levels):
                    elapsed = time.perf_counter() - started
                    percent = output_id / len(levels)
                    eta = elapsed * (1 - percent) / percent
                    filled = round(30 * percent)
                    print(f'\r[{"#" * filled}{"-" * (30 - filled)}] '
                          f'{output_id}/{len(levels)} ({100 * percent:5.1f}%) '
                          f'ETA {_format_seconds(eta)}', end='', flush=True)
                offset = ((offset + len(record) + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        print()
        db.execute('INSERT INTO DiaFrameMsMsWindowGroups(Id) VALUES (1)')
        db.execute('INSERT INTO DiaFrameMsMsWindows(WindowGroup, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth, CollisionEnergy) VALUES (1,0,918,900.0,1610.0,35.0)')
        db.execute('UPDATE Segments SET FirstFrame=1, LastFrame=?', (len(levels),))
        db.commit()
    finally:
        db.close()
        reader.mbi.Close()
    for key, value in {'input_frame_count': len(levels) + 1, 'skipped_empty_input_frames': [0],
                       'output_frame_count': len(levels), 'output_ms1_frame_count': ms1,
                       'output_ms2_frame_count': ms2, 'output_peak_count': total_peaks,
                       'timing_sdk_read_seconds': round(read_seconds, 3),
                       'timing_map_seconds': round(map_seconds, 3),
                       'timing_encode_seconds': round(encode_seconds, 3),
                       'timing_write_seconds': round(write_seconds, 3)}.items():
        logger.info('%s: %s', key, value)

def _resolve_jobs(input_path, output_arg, parser):
    if input_path.is_file():
        if input_path.suffix.lower() != '.mbi':
            parser.error(f'Input must be a .mbi file: {input_path}')
        output = output_arg if output_arg else input_path.with_suffix('.d')
        return [(input_path, output)]
    if input_path.is_dir():
        inputs = sorted(input_path.glob('*.mbi'))
        if not inputs:
            parser.error(f'No .mbi files found in: {input_path}')
        output_dir = output_arg if output_arg else input_path
        if output_dir.suffix.lower() == '.d':
            parser.error('For a folder input, --output must be an output folder, not a .d path.')
        return [(item, output_dir / f'{item.stem}.d') for item in inputs]
    parser.error(f'Input path not found: {input_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='xtracer convert', description='Convert PAMAF .mbi files to Bruker TDF .d directories.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('input_mbi', nargs='?', type=Path, help='One input .mbi file.')
    source.add_argument('-ws_in', type=Path, help='Folder containing .mbi files for one batch conversion run.')
    parser.add_argument('-o', '--output', type=Path, help='Output .d path for a single input file.')
    parser.add_argument('-out_name', default='diann_diatracer', help='Output folder name below -ws_in.')
    parser.add_argument('--force', action='store_true', help='Delete and recreate outputs that already exist.')
    args = parser.parse_args(argv)
    if args.ws_in:
        ws_in = args.ws_in.resolve()
        if not ws_in.is_dir():
            parser.error(f'Input folder not found: {ws_in}')
        inputs = sorted(ws_in.glob('*.mbi'))
        if not inputs:
            parser.error(f'No .mbi files found in: {ws_in}')
        output_root = ws_in / args.out_name
        jobs = [(item, output_root / f'{item.stem}.d') for item in inputs]
        batch = True
    else:
        input_mbi = args.input_mbi.resolve()
        if not input_mbi.is_file() or input_mbi.suffix.lower() != '.mbi':
            parser.error(f'Input .mbi file not found: {input_mbi}')
        output = args.output.resolve() if args.output else input_mbi.with_suffix('.d')
        output_root, jobs, batch = output.parent, [(input_mbi, output)], False
    output_root.mkdir(parents=True, exist_ok=True)
    parameters = {'ws_in': args.ws_in.resolve() if args.ws_in else None,
                  'input_mbi': args.input_mbi.resolve() if args.input_mbi else None,
                  'output_root': output_root, 'out_name': args.out_name if batch else None,
                  'input_mbi_count': len(jobs), 'force': args.force, 'profile': PROFILE,
                  'synthetic_ion_mobility': True, 'num_scans': NUM_SCANS,
                  'scan_mapping': 'scan = round(917 - 917/400 * AT_ms)',
                  'one_over_k0_mapping': '1/K0 = 0.5 + 0.0032 * AT_ms',
                  'tof_mapping': 'quadratic template calibration', 'tof_max_index': TIMS_MAX_TOF,
                  'binary_record_format': 'type_2', 'binary_compression': 'zstd_level_1',
                  'binary_alignment_bytes': ALIGNMENT, 'dia_window': [0, 918, 900.0, 1610.0, 35.0]}
    Logger.set_logger(output_root, run_name='xtracer_convert',
                      command='xtracer convert ' + ' '.join(sys.argv[1:] if argv is None else argv),
                      parameters=parameters)
    logger = Logger.get_logger()
    failures = skipped = 0
    for index, (input_mbi, output) in enumerate(jobs, start=1):
        if output.exists():
            if not args.force:
                logger.info('skip %s/%s: output exists: %s', index, len(jobs), output)
                skipped += 1
                continue
            shutil.rmtree(output)
        logger.info('start %s/%s: %s', index, len(jobs), input_mbi.name)
        print(f'\\n[{index}/{len(jobs)}] {input_mbi.name}', flush=True)
        try:
            _convert(input_mbi, output, logger)
            logger.info('success %s/%s: %s', index, len(jobs), output)
        except Exception:
            if output.exists():
                shutil.rmtree(output)
            logger.exception('failed %s/%s: %s', index, len(jobs), input_mbi)
            failures += 1
    logger.info('status: %s; converted=%s skipped=%s failed=%s',
                'success' if not failures else 'failed', len(jobs)-skipped-failures, skipped, failures)
    return 1 if failures else 0


if __name__ == '__main__':
    main()