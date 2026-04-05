import argparse
from pathlib import Path
from xtracer import search
from xtracer.log import Logger

def parse_args():
    parser = argparse.ArgumentParser('xTracer')

    # required=True
    parser.add_argument(
        '-ws_in', required=True,
        help='Specify the folder that contains .mbi or .d files.'
    )
    parser.add_argument(
        '-out_name', type=str, default='mgf_xtracer',
        help='Specify the output folder name where the mgf will be saved.'
    )
    parser.add_argument(
        '-data_type', required=True,
        help='Specify the type of ms file. PAMAF or diaPASEF are supported.'
    )
    # optional
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-xic_only', action='store_true', help='XIC-based PCC')
    mode_group.add_argument(
        '-xic_xim', action='store_true', help='XIC+XIM averaged PCC'
    )
    parser.add_argument('-write_pcc',
                        action='store_true',
                        help='Write PCC values or not')

    # common params
    parser.add_argument(
        '-pr_mz_min', type=float, default=200,
        help='Specify the minimum m/z value of precursors. Default: 200'
    )
    parser.add_argument(
        '-charge_min', type=int, default=2,
        help='Specify the minimum charge of precursors. Default: 1'
    )
    parser.add_argument(
        '-charge_max', type=int, default=4,
        help='Specify the maximum charge of precursors. Default: 4'
    )
    parser.add_argument(
        '-tol_im_area', type=float, default=0.02,
        help='Specify the millisecond tolerance of signal in at dimension. '
             'Default: 0.02'
    )
    parser.add_argument(
        '-tol_im_shift', type=float, default=0.01,
        help='Specify the millisecond tolerance when considering signal '
             'related. Default: 0.01'
    )
    parser.add_argument(
        '-tol_ppm', type=float, default=20,
        help='Specify the ppm tolerance of signal in m/z dimension. '
             'Default: 20'
    )
    parser.add_argument(
        '-tol_iso_num', type=int, default=1,
        help='Specify how many isotopes should have to be a precursor. '
             'Default: 1, i.e. M and M+1'
    )
    parser.add_argument(
        '-tol_pcc', type=float, default=0.3,
        help='Specify the PCC tolerance when two signal are related. Default: 0.3'
    )
    parser.add_argument(
        '-tol_point_num', type=int, default=5,
        help='Specify the point num tolerance that a signal should have. '
             'Default: 5'
    )
    parser.add_argument(
        '-tol_fg_num', type=int, default=10,
        help='Specify the fragment ions num tolerance that a spectrum should '
             'have. Default: 10'
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

    MODE_MAP = {"xic_only": args.xic_only, "xic_xim": args.xic_xim}
    run_mode = [k for k, v in MODE_MAP.items() if v][0]

    if args.data_type.lower() == 'pamaf':
        fin_v = list(Path(args.ws_in).glob('*.mbi'))
    elif args.data_type.lower() == 'diapasef':
        fin_v = [x for x in Path(args.ws_in).glob('*.d') if x.is_dir()]
    else:
        raise ValueError(f'Invalid data type {args.data_type}, should be [pamaf, diapasef]')

    outdir = args.ws_in / args.out_name
    outdir.mkdir(exist_ok=True)
    Logger.set_logger(outdir)
    logger = Logger.get_logger()
    logger.info('xTracer, for SLIM with high resolution ion mobility')
    for fi, fin in enumerate(fin_v[::-1]):
        logger.info(f'Processing {fi+1}/{len(fin_v)} in {run_mode} mode: {fin.name}')
        fout = outdir / (fin.stem + '.mgf')
        search.main(args, fin, fout, run_mode)


if __name__ == '__main__':
    main()
