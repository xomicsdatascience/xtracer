import argparse
from pathlib import Path
from xtracer import xic, xim, xix
from xtracer.log import Logger

def parse_args():
    parser = argparse.ArgumentParser('MobiTracer')

    # required=True
    parser.add_argument(
        '-ws_in', required=True,
        help='Specify the folder that contains .mbi files.'
    )
    parser.add_argument(
        '-out_name', type=str, default='mgf_xtracer',
        help='Specify the folder name that contains .mgf files.'
    )

    # optional
    parser.add_argument('-xic',
                        action='store_true',
                        help='Using XIC-based method')
    parser.add_argument('-xim',
                        action='store_true',
                        help='Using XIM-based method')

    # common params
    parser.add_argument(
        '-pr_mz_min', type=float, default=200,
        help='Specify the minimum m/z value of precursors. Default: 200'
    )
    parser.add_argument(
        '-charge_min', type=int, default=1,
        help='Specify the minimum charge of precursors. Default: 1'
    )
    parser.add_argument(
        '-charge_max', type=int, default=4,
        help='Specify the maximum charge of precursors. Default: 4'
    )
    parser.add_argument(
        '-at_min', type=float, default=90,
        help='Specify the minimum at value of signals. Default: 90'
    )
    parser.add_argument(
        '-tol_at_area', type=float, default=2.5,
        help='Specify the millisecond tolerance of signal in at dimension. '
             'Default: 2.5'
    )
    parser.add_argument(
        '-tol_at_shift', type=float, default=1,
        help='Specify the millisecond tolerance when considering signal '
             'related. Default: 1'
    )
    parser.add_argument(
        '-tol_ppm', type=float, default=30,
        help='Specify the ppm tolerance of signal in m/z dimension. '
             'Default: 30'
    )
    parser.add_argument(
        '-tol_iso_num', type=int, default=2,
        help='Specify how many isotopes should have to be a precursor. '
             'Default: 2, i.e. M, M+1H, M+2H'
    )
    parser.add_argument(
        '-tol_pcc', type=float, default=0.3,
        help='Specify the PCC tolerance when two signal are related. Default: 0.3'
    )
    parser.add_argument(
        '-tol_point_num', type=int, default=9,
        help='Specify the point num tolerance that a signal should have. '
             'Default: 9'
    )
    parser.add_argument(
        '-tol_fg_num', type=int, default=8,
        help='Specify the fragment ions num tolerance that a spectrum should '
             'have. Default: 8'
    )

    # for xim
    parser.add_argument(
        '-xim_across_cycle_num', type=int, default=3,
        help='Specify the odd XIM cycle span when summing frames. Default: 3'
    )

    # for xic
    parser.add_argument(
        '-xic_across_cycle_num', type=int, default=7,
        help='Specify the odd XIC cycle span when extracting XIC. Default: 7'
    )

    # process params
    args = parser.parse_args()
    args.ws_in = Path(args.ws_in)
    return args


def main():
    args = parse_args()

    fin_v = list(Path(args.ws_in).glob('*.mbi'))

    if args.xim and not args.xic:
        outdir = args.ws_in / args.out_name
        outdir.mkdir(exist_ok=True)
        Logger.set_logger(outdir)
        logger = Logger.get_logger()
        logger.info('MobiTracer, for SLIM with high resolution ion mobility')
        for fi, fin in enumerate(fin_v):
            logger.info(f'Processing {fi+1}/{len(fin_v)}')
            fout = outdir / (fin.stem + '.mgf')
            xim.main(args, fin, fout)
    elif args.xic and not args.xim:
        outdir = args.ws_in / args.out_name
        outdir.mkdir(exist_ok=True)
        Logger.set_logger(outdir)
        logger = Logger.get_logger()
        logger.info('MobiTracer, for SLIM with high resolution ion mobility')
        for fi, fin in enumerate(fin_v):
            logger.info(f'Processing {fi+1}/{len(fin_v)}')
            fout = outdir / (fin.stem + '.mgf')
            xic.main(args, fin, fout)
    elif args.xim and args.xic:
        outdir = args.ws_in / args.out_name
        outdir.mkdir(exist_ok=True)
        Logger.set_logger(outdir)
        logger = Logger.get_logger()
        logger.info('MobiTracer, for SLIM with high resolution ion mobility')
        for fi, fin in enumerate(fin_v):
            logger.info(f'Processing {fi + 1}/{len(fin_v)}')
            fout = outdir / (fin.stem + '.mgf')
            xix.main(args, fin, fout)
    else:
        raise Exception('You must specify either -xim or -xic')


if __name__ == '__main__':
    main()
