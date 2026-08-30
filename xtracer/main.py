import argparse
from pathlib import Path
from xtracer import search
from xtracer.log import Logger

def parse_args():
    parser = argparse.ArgumentParser('xTracer')

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
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-xic', action='store_true', help='XIC-based PCC')
    mode_group.add_argument('-xim', action='store_true', help='XIM-based PCC')
    mode_group.add_argument(
        '-xix', action='store_true', help='XIC+XIM averaged PCC'
    )
    parser.add_argument('-write_pcc',
                        action='store_true',
                        help='Write PCC values or not')
    parser.add_argument('-xic_mid_int',
                        action='store_true',
                        help='Use XIC center-frame intensity for fragment '
                             'peaks instead of merged-frame point height')
    parser.add_argument('-merge_weighted',
                        action='store_true',
                        help='Intensity-weighted average when merging '
                             'frames (default: take mz/at of the stronger '
                             'point)')
    parser.add_argument(
        '-merge_at_tol', type=float, default=0.001,
        help='Arrival-time tolerance in ms when merging coincident points '
             'across frames. Default: 0.001'
    )
    parser.add_argument('-allow_lone',
                        action='store_true',
                        help='Allow precursors without visible isotopes '
                             '(Gaussian-shaped XIC/XIM rescue)')
    parser.add_argument('-apex_only',
                        action='store_true',
                        help='Emit spectra only at the apex frame of each '
                             'precursor XIC (removes cross-frame redundancy)')
    parser.add_argument('-iso_rescue',
                        action='store_true',
                        help='Rescue precursors whose M+1 isotope is not a '
                             'local-maximum seed by extracting an XIC '
                             'directly at the theoretical isotope m/z')
    parser.add_argument(
        '-iso_rescue_pcc', type=float, default=0.3,
        help='Min PCC between seed XIC and rescued isotope XIC. Default: 0.3'
    )
    parser.add_argument(
        '-iso_rescue_gauss', type=float, default=0.6,
        help='Min PCC of rescued isotope XIC against Gaussian shape. '
             'Default: 0.6'
    )
    parser.add_argument('-consensus',
                        action='store_true',
                        help='Merge spectra of the same precursor across '
                             'cycles into one consensus spectrum per '
                             'precursor-charge (richer peaks, less redundancy)')

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
        '-at_min', type=float, default=100,
        help='Specify the minimum at value of signals. Default: 100'
    )
    parser.add_argument(
        '-tol_at_area', type=float, default=2.0,
        help='Specify the millisecond tolerance of signal in at dimension. '
             'Default: 2.0'
    )
    parser.add_argument(
        '-tol_at_shift', type=float, default=1.0,
        help='Specify the millisecond tolerance when considering signal '
             'related. Default: 1.0'
    )
    parser.add_argument(
        '-tol_ppm', type=float, default=30,
        help='Specify the ppm tolerance of signal in m/z dimension. '
             'Default: 30'
    )
    parser.add_argument(
        '-tol_iso_num', type=int, default=1,
        help='Specify how many isotopes should have to be a precursor. '
             'Default: 1, i.e. M, M+1'
    )
    parser.add_argument(
        '-iso_int_max', type=float, default=1.0,
        help='Specify the max intensity ratio of M+N relative to M when '
             'validating isotope clusters. Default: 1.0'
    )
    parser.add_argument(
        '-iso_int_min', type=float, default=0.0,
        help='Specify the min intensity ratio of M+N relative to M when '
             'validating isotope clusters. Default: 0.0 (no lower bound)'
    )
    parser.add_argument(
        '-tol_pcc', type=float, default=0.3,
        help='Specify the PCC tolerance when two signal are related. Default: 0.3'
    )
    parser.add_argument(
        '-tol_neighbor1_num', type=int, default=5,
        help='Specify the neighbor num tolerance that a MS1 signal should have. '
             'Default: 5'
    )
    parser.add_argument(
        '-tol_neighbor2_num', type=int, default=3,
        help='Specify the neighbor num tolerance that a MS2 signal should have. '
             'Default: 3'
    )
    parser.add_argument(
        '-tol_fg_num', type=int, default=10,
        help='Specify the fragment ions num tolerance that a spectrum should '
             'have. Default: 10'
    )
    parser.add_argument(
        '-tol_pcc_strong', type=float, default=0.6,
        help='PCC threshold defining a strongly-correlated fragment match. '
             'Default: 0.6'
    )
    parser.add_argument(
        '-tol_fg_num_strong', type=int, default=0,
        help='Require at least this many strong (PCC>tol_pcc_strong) fragment '
             'matches per spectrum; 0 disables the gate. Default: 0'
    )
    parser.add_argument(
        '-frag_ppm_shift', type=float, default=0.0,
        help='Shift fragment m/z by this ppm when writing spectra, to '
             'compensate systematic frame2 mass bias. Default: 0.0'
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

    MODE_MAP = {"xic": args.xic, "xim": args.xim, "xix": args.xix}
    run_mode = [k for k, v in MODE_MAP.items() if v][0]

    fin_v = list(Path(args.ws_in).glob('*.mbi'))

    outdir = args.ws_in / args.out_name
    outdir.mkdir(exist_ok=True)
    Logger.set_logger(outdir)
    logger = Logger.get_logger()
    logger.info('xTracer, for SLIM with high resolution ion mobility')
    logger.info(vars(args))
    for fi, fin in enumerate(fin_v):
        logger.info(f'Processing {fi+1}/{len(fin_v)} in {run_mode} mode')
        fout = outdir / (fin.stem + '.mgf')
        search.main(args, fin, fout, run_mode)


if __name__ == '__main__':
    main()
