import argparse
import sys
from pathlib import Path

from xtracer.log import Logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        'xTracer',
        description='Generate PAMAF pseudo-spectra.',
        epilog='Additional commands: xtracer convert --help; xtracer gui --help',
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
    return args


def run_pseudo(argv=None):
    args = parse_args(argv)
    mode_map = {'xic': args.xic, 'xim': args.xim, 'xix': args.xix}
    run_mode = next(name for name, enabled in mode_map.items() if enabled)
    outdir = args.ws_in / args.out_name
    parameters = vars(args).copy()
    parameters['mode'] = run_mode
    Logger.set_logger(
        outdir,
        run_name='xtracer',
        command='xtracer ' + ' '.join(sys.argv[1:] if argv is None else argv),
        parameters=parameters,
    )
    logger = Logger.get_logger()
    inputs = list(args.ws_in.glob('*.mbi'))
    logger.info('xTracer, for SLIM with high resolution ion mobility')
    logger.info('input_mbi_count: %s', len(inputs))
    if not inputs:
        logger.warning('No .mbi files found in %s', args.ws_in.resolve())
        logger.info('status: success')
        return 0

    try:
        # Keep help and non-MBI commands usable without the private vendor SDK.
        from xtracer import search
        for index, input_mbi in enumerate(inputs, start=1):
            logger.info('Processing %s/%s in %s mode', index, len(inputs), run_mode)
            output_mgf = outdir / f'{input_mbi.stem}.mgf'
            search.main(args, input_mbi, output_mgf, run_mode)
            logger.info('output_mgf: %s', output_mgf.resolve())
    except Exception:
        logger.exception('status: failed')
        raise
    logger.info('status: success')
    return 0


def main(argv=None):
    """Dispatch pseudo-spectrum generation and the xTracer command family."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'convert':
        from xtracer.convert import main as convert_main
        return convert_main(argv[1:])
    if argv and argv[0] == 'gui':
        from xtracer.gui import main as gui_main
        return gui_main(argv[1:])
    return run_pseudo(argv)


if __name__ == '__main__':
    main()
