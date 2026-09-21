"""CLI wrapper for the Streamlit Sage-result viewer."""

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

from xtracer.log import Logger


def normalize_sage_filename(filename):
    """Return the local basename represented by a Sage filename field."""
    return Path(unquote(str(filename))).name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='xtracer gui',
        description='Launch the xTracer viewer for Sage-annotated pseudo-spectra.',
    )
    parser.add_argument('--mbi', required=True, type=Path, help='Input .mbi file.')
    parser.add_argument('--mgf', required=True, type=Path, help='xTracer pseudo-spectrum .mgf file.')
    parser.add_argument('--sage-results', required=True, type=Path, help='Sage results.sage.tsv file.')
    parser.add_argument('--matched-fragments', type=Path, help='Sage matched_fragments.sage.tsv file.')
    parser.add_argument('--out-dir', required=True, type=Path, help='Directory for the GUI launch log.')
    args = parser.parse_args(argv)
    if args.matched_fragments is None:
        args.matched_fragments = args.sage_results.parent / 'matched_fragments.sage.tsv'
    return args


def main(argv=None):
    args = parse_args(argv)
    Logger.set_logger(
        args.out_dir,
        run_name='xtracer-gui',
        command='xtracer gui ' + ' '.join(sys.argv[1:] if argv is None else argv),
        parameters=vars(args),
    )
    logger = Logger.get_logger()
    missing = [path for path in (args.mbi, args.mgf, args.sage_results, args.matched_fragments) if not path.is_file()]
    if missing:
        logger.error('status: failed; missing input: %s', ', '.join(map(str, missing)))
        Logger.close()
        return 2

    script = Path(__file__).with_name('streamlit_sage.py').resolve()
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', str(script),
        '--server.headless', 'false', '--',
        str(args.mbi.resolve()), str(args.mgf.resolve()),
        str(args.sage_results.resolve()), str(args.matched_fragments.resolve()),
    ]
    logger.info('streamlit_command: %s', ' '.join(cmd))
    try:
        completed = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        logger.info('status: stopped by user')
        Logger.close()
        return 0
    except OSError as exc:
        logger.exception('status: failed; unable to start Streamlit: %s', exc)
        Logger.close()
        return 1
    if completed.returncode:
        logger.error('status: failed; Streamlit exit_code=%s', completed.returncode)
        Logger.close()
        return completed.returncode
    logger.info('status: success')
    Logger.close()
    return 0
