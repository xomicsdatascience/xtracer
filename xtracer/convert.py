"""Convert PAMAF .mbi files to single-window Bruker TDF .d datasets."""
import argparse
import shutil
import sqlite3
import sys
import time
import uuid
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



FRAME_COLUMNS = ('Id', 'Time', 'Polarity', 'ScanMode', 'MsMsType', 'TimsId',
                 'MaxIntensity', 'SummedIntensities', 'NumScans', 'NumPeaks',
                 'MzCalibration', 'T1', 'T2', 'TimsCalibration', 'PropertyGroup',
                 'AccumulationTime', 'RampTime')


def _create_tdf(tdf_path, input_name):
    """Create a self-contained, minimal TDF SQLite database."""
    db = sqlite3.connect(tdf_path)
    db.executescript("""
        PRAGMA page_size=4096;
        CREATE TABLE DiaFrameMsMsInfo (Frame INTEGER PRIMARY KEY, WindowGroup INTEGER NOT NULL);
        CREATE TABLE DiaFrameMsMsWindowGroups (Id INTEGER PRIMARY KEY);
        CREATE TABLE DiaFrameMsMsWindows (WindowGroup INTEGER NOT NULL, ScanNumBegin INTEGER NOT NULL, ScanNumEnd INTEGER NOT NULL, IsolationMz REAL NOT NULL, IsolationWidth REAL NOT NULL, CollisionEnergy REAL NOT NULL, PRIMARY KEY (WindowGroup, ScanNumBegin)) WITHOUT ROWID;
        CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL NOT NULL, Polarity TEXT NOT NULL, ScanMode INTEGER NOT NULL, MsMsType INTEGER NOT NULL, TimsId INTEGER NOT NULL, MaxIntensity INTEGER NOT NULL, SummedIntensities INTEGER NOT NULL, NumScans INTEGER NOT NULL, NumPeaks INTEGER NOT NULL, MzCalibration INTEGER NOT NULL, T1 REAL NOT NULL, T2 REAL NOT NULL, TimsCalibration INTEGER NOT NULL, PropertyGroup INTEGER, AccumulationTime REAL NOT NULL, RampTime REAL NOT NULL);
        CREATE UNIQUE INDEX FramesTimeIndex ON Frames(Time);
        CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT NOT NULL);
        CREATE TABLE MzCalibration (Id INTEGER PRIMARY KEY, ModelType INTEGER NOT NULL, DigitizerTimebase REAL NOT NULL, DigitizerDelay REAL NOT NULL, T1 REAL NOT NULL, T2 REAL NOT NULL, dC1 REAL NOT NULL, dC2 REAL NOT NULL, C0 REAL NOT NULL, C1 REAL NOT NULL, C2 REAL NOT NULL, C3 REAL NOT NULL, C4 REAL NOT NULL);
        CREATE TABLE Segments (Id INTEGER PRIMARY KEY, FirstFrame INTEGER NOT NULL, LastFrame INTEGER NOT NULL, IsCalibrationSegment INTEGER NOT NULL);
        CREATE TABLE TimsCalibration (Id INTEGER PRIMARY KEY, ModelType INTEGER NOT NULL, C0 REAL NOT NULL, C1 REAL NOT NULL, C2 REAL NOT NULL, C3 REAL NOT NULL, C4 REAL NOT NULL, C5 REAL NOT NULL, C6 REAL NOT NULL, C7 REAL NOT NULL, C8 REAL NOT NULL, C9 REAL NOT NULL);
    """)
    metadata = [
        ('SchemaType', 'TDF'), ('SchemaVersionMajor', '3'), ('SchemaVersionMinor', '4'),
        ('ClosedProperly', '1'), ('TimsCompressionType', '2'), ('MaxNumPeaksPerScan', '0'),
        ('AnalysisId', str(uuid.uuid4())), ('DigitizerNumSamples', str(TIMS_MAX_TOF)),
        ('MzAcqRangeLower', '100.0'), ('MzAcqRangeUpper', '1700.0'),
        ('AcquisitionSoftwareVendor', 'xTracer'), ('AcquisitionSoftware', 'xTracer'),
        ('AcquisitionSoftwareVersion', PROFILE), ('InstrumentVendor', 'PAMAF'),
        ('InstrumentName', 'PAMAF synthetic TDF'),
        ('Description', 'Synthetic TDF-compatible representation generated from MBI.'),
        ('SampleName', input_name), ('PeakListIndexScaleFactor', '1'),
        ('OneOverK0AcqRangeLower', '0.5'), ('OneOverK0AcqRangeUpper', '1.78'),
    ]
    db.executemany('INSERT INTO GlobalMetadata(Key, Value) VALUES (?, ?)', metadata)
    db.execute('INSERT INTO MzCalibration VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
               (1, 1, MZ_TIMEBASE, MZ_DELAY, 0.0, 0.0, 0.0, 0.0,
                MZ_C0, MZ_C1, 0.0, 0.0, 0.0))
    db.execute('INSERT INTO TimsCalibration VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
               (1, 2, *TIMS_CALIBRATION))
    db.execute('INSERT INTO Segments VALUES (1, 1, 1, 0)')
    return db

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


def _insert_frame(db, frame_id, rt, msms_type, tims_id, peak_count, max_intensity, summed_intensities):
    values = (frame_id, float(rt), '+', 9, msms_type, int(tims_id),
              int(max_intensity), int(summed_intensities), NUM_SCANS,
              int(peak_count), 1, 25.617137181405713, 26.549323743070385,
              1, None, 99.953, 99.953)
    db.execute(f'INSERT INTO Frames ({",".join(FRAME_COLUMNS)}) VALUES ({",".join("?" for _ in FRAME_COLUMNS)})', values)

def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02}:{(seconds % 3600) // 60:02}:{seconds % 60:02}'


def _convert(input_mbi, output, logger):
    from xtracer.mbi import MBIReader
    output.mkdir(parents=True)
    reader = MBIReader(input_mbi, 2)
    levels = np.asarray(reader.GetFrameMSLevels(), dtype=np.int64)
    rts = np.asarray(reader.GetRetentionTimes(), dtype=np.float64)
    if levels.size != rts.size or levels.size < 2:
        raise RuntimeError('Invalid MBI frame metadata.')
    levels, rts = levels[1:], rts[1:]
    unsupported = sorted(set(levels) - set(LEVEL_MSMSTYPE))
    if unsupported:
        raise RuntimeError(f'Unsupported MBI MS levels: {unsupported}')
    db = _create_tdf(output / 'analysis.tdf', input_mbi.name)
    ms1 = ms2 = total_peaks = 0
    max_peaks_per_scan = 0
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
                _insert_frame(db, output_id, rt, msms_type, offset,
                              len(tofs), int(encoded.max()) if encoded.size else 0,
                              int(encoded.sum(dtype=np.uint64)))
                max_peaks_per_scan = max(max_peaks_per_scan, int(counts.max()))
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
        db.execute("UPDATE GlobalMetadata SET Value=? WHERE Key='MaxNumPeaksPerScan'",
                   (str(max_peaks_per_scan),))
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

def main(argv=None):
    parser = argparse.ArgumentParser(prog='xtracer convert', description='Convert PAMAF .mbi files to Bruker TDF .d directories.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('input_mbi', nargs='?', type=Path, help='One input .mbi file.')
    source.add_argument('-ws_in', type=Path, help='Folder containing .mbi files for one batch conversion run.')
    parser.add_argument('-o', '--output', type=Path, help='Output .d path for a single input file.')
    parser.add_argument('-out_name', default='mbi2d', help='Output folder name below -ws_in.')
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
                  'tof_mapping': 'quadratic TDF calibration', 'tof_max_index': TIMS_MAX_TOF,
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
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        logger.info('start %s/%s: %s', index, len(jobs), input_mbi.name)
        print(f'\n[{index}/{len(jobs)}] {input_mbi.name}', flush=True)
        try:
            _convert(input_mbi, output, logger)
            logger.info('success %s/%s: %s', index, len(jobs), output)
        except Exception:
            if output.exists():
                if output.is_dir():
                    shutil.rmtree(output)
                else:
                    output.unlink()
            logger.exception('failed %s/%s: %s', index, len(jobs), input_mbi)
            failures += 1
    logger.info('status: %s; converted=%s skipped=%s failed=%s',
                'success' if not failures else 'failed', len(jobs)-skipped-failures, skipped, failures)
    Logger.close()
    return 1 if failures else 0


if __name__ == '__main__':
    main()
