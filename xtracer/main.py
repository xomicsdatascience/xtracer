import argparse
import sys
from pathlib import Path

from xtracer.log import Logger


def parse_search_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='xtracer search',
        description='Generate PAMAF pseudo-spectra.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-ws_in', required=True, help='Folder containing .mbi files.')
    parser.add_argument('-out_name', type=str, default='mgf_xtracer', help='MGF output folder name.')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-xic', action='store_true', help='XIC-based PCC')
    mode_group.add_argument('-xim', action='store_true', help='XIM-based PCC')
    mode_group.add_argument('-xix', action='store_true', help='XIC+XIM averaged PCC')
    parser.add_argument('-write_pcc', action='store_true', help='Write PCC values to MS/MS files')
    parser.add_argument('-pr_mz_min', type=float, default=200, help='Minimum precursor m/z.')
    parser.add_argument('-charge_min', type=int, default=2, help='Minimum precursor charge.')
    parser.add_argument('-charge_max', type=int, default=4, help='Maximum precursor charge.')
    parser.add_argument('-at_min', type=float, default=100, help='Minimum arrival time in ms.')
    parser.add_argument('-tol_at_area', type=float, default=2.0, help='Arrival-time integration tolerance in ms.')
    parser.add_argument('-tol_at_shift', type=float, default=1.0, help='Arrival-time matching tolerance in ms.')
    parser.add_argument('-tol_ppm', type=float, default=30, help='m/z matching tolerance in ppm.')
    parser.add_argument('-tol_iso_num', type=int, default=2, help='Required isotope count to the right of M.')
    parser.add_argument('-tol_pcc', type=float, default=0.3, help='Minimum precursor-fragment PCC.')
    parser.add_argument('-tol_neighbor1_num', type=int, default=5, help='MS1 local-neighbor threshold.')
    parser.add_argument('-tol_neighbor2_num', type=int, default=3, help='MS2 local-neighbor threshold.')
    parser.add_argument('-tol_fg_num', type=int, default=10, help='Minimum matched fragment count.')
    parser.add_argument('-xim_across_cycle_num', type=int, default=3, help='Odd XIM cycle span.')
    parser.add_argument('-xic_across_cycle_num', type=int, default=7, help='Odd XIC cycle span.')
    args = parser.parse_args(argv)
    args.ws_in = Path(args.ws_in)
    if not args.ws_in.is_dir():
        parser.error(f'Input folder not found: {args.ws_in}')
    if args.charge_min < 1 or args.charge_max < args.charge_min:
        parser.error('Require 1 <= -charge_min <= -charge_max')
    if args.tol_iso_num < 1:
        parser.error('-tol_iso_num must be at least 1')
    for name in ('tol_at_area', 'tol_at_shift', 'tol_ppm'):
        if getattr(args, name) <= 0:
            parser.error(f'-{name} must be positive')
    for name in ('tol_neighbor1_num', 'tol_neighbor2_num', 'tol_fg_num'):
        if getattr(args, name) < 1:
            parser.error(f'-{name} must be at least 1')
    if not -1 <= args.tol_pcc <= 1:
        parser.error('-tol_pcc must be between -1 and 1')
    if args.xim_across_cycle_num < 3 or args.xim_across_cycle_num % 2 != 1:
        parser.error('-xim_across_cycle_num must be an odd integer of at least 3')
    if args.xic_across_cycle_num < 3 or args.xic_across_cycle_num % 2 != 1:
        parser.error('-xic_across_cycle_num must be an odd integer of at least 3')
    if (args.xic or args.xix) and args.xic_across_cycle_num < args.xim_across_cycle_num:
        parser.error('-xic_across_cycle_num must be at least -xim_across_cycle_num')
    return args


def run_search(argv=None):
    args = parse_search_args(argv)
    mode_map = {'xic': args.xic, 'xim': args.xim, 'xix': args.xix}
    run_mode = next(name for name, enabled in mode_map.items() if enabled)
    outdir = args.ws_in / args.out_name
    parameters = vars(args).copy()
    parameters['mode'] = run_mode
    Logger.set_logger(
        outdir,
        run_name='xtracer_search',
        command='xtracer search ' + ' '.join(
            sys.argv[2:] if argv is None else argv
        ),
        parameters=parameters,
    )
    logger = Logger.get_logger()
    inputs = sorted(args.ws_in.glob('*.mbi'))
    logger.info('xTracer, for SLIM with high resolution ion mobility')
    logger.info('input_mbi_count: %s', len(inputs))
    if not inputs:
        logger.error('status: failed; no .mbi files found in %s', args.ws_in.resolve())
        Logger.close()
        return 2

    try:
        from xtracer import search
        for index, input_mbi in enumerate(inputs, start=1):
            logger.info('Processing %s/%s in %s mode', index, len(inputs), run_mode)
            output_mgf = outdir / f'{input_mbi.stem}.mgf'
            search.main(args, input_mbi, output_mgf, run_mode)
            logger.info('output_mgf: %s', output_mgf.resolve())
    except Exception:
        logger.exception('status: failed')
        Logger.close()
        raise
    logger.info('status: success')
    Logger.close()
    return 0


def _root_parser():
    parser = argparse.ArgumentParser(
        prog='xtracer',
        description='PAMAF pseudo-spectrum generation, conversion, and visualization.',
    )
    commands = parser.add_subparsers(dest='command', metavar='{search,convert,gui}')
    commands.add_parser('search', add_help=False, help='Generate pseudo-spectra from .mbi files.')
    commands.add_parser('convert', add_help=False, help='Convert .mbi files to Bruker TDF .d directories.')
    commands.add_parser('gui', add_help=False, help='Launch the Sage-result viewer.')
    return parser


def main(argv=None):
    """Dispatch the xTracer command family."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _root_parser()
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    if argv[0] == 'search':
        return run_search(argv[1:])
    if argv[0] == 'convert':
        from xtracer.convert import main as convert_main
        return convert_main(argv[1:])
    if argv[0] == 'gui':
        from xtracer.gui import main as gui_main
        return gui_main(argv[1:])
    parser.error(f'unknown command: {argv[0]}')


if __name__ == '__main__':
    raise SystemExit(main())
