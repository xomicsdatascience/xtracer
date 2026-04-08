import numpy as np

import numba.typed
from xtracer.utils import *
from xtracer.mbi import MBIReader
from xtracer.tims import Tims
from io import StringIO

try:
    profile
except:
    profile = lambda x: x


def get_points(xics1_v, ats1_v, mzs1_v, ints1_v, cycle1_v):
    xics1 = np.vstack(xics1_v)
    ats1 = np.concatenate(ats1_v)
    mzs1 = np.concatenate(mzs1_v)
    ints1 = np.concatenate(ints1_v)
    cycle1 = np.concatenate(cycle1_v)
    idx = np.argsort(mzs1)
    mzs = mzs1[idx]
    ats = ats1[idx]
    ints = ints1[idx]
    cycles = cycle1[idx]
    xics = xics1[idx]
    return cycles, ats, mzs, ints, xics

@profile
def main(args, indir, outdir, mode):
    # read
    across_cycle_num = args.xic_across_cycle_num

    if indir.suffix == '.mbi':
        ms = MBIReader(indir, across_cycle_num)
    elif indir.suffix == '.d':
        ms = Tims(indir, across_cycle_num)

    with open(outdir, "w") as f:
        buffer = StringIO()
        buffer_write = buffer.write
        pseudo_count = 1
        quad_num = ms.get_quad_num()
        quads = ms.get_dia_quadrupole()
        for quad_idx in range(1, 1+quad_num):
            quad = quads[quad_idx:(quad_idx+1)]
            # loop cycle
            cycle_rts = ms.get_frame_times(quad_idx)

            xics1_v, xics2_v = [], []
            ats1_v, mzs1_v, ints1_v, cycle1_v = [], [], [], []
            ats2_v, mzs2_v, ints2_v, cycle2_v = [], [], [], []
            state_m_v = []
            for cycle_i in range(3, len(cycle_rts) - 3):
                # cycle_i = 203

                # load frames
                ms.load_cycles_to_deque(quad_idx, cycle_i)
                frame1_deque = ms.deque_frame1
                frame2_deque = ms.deque_frame2

                # merge frames for maximum points
                frame1_at, frame1_mz, frame1_height = merge_frames(
                    ms.deque_frame1, 3)
                frame2_at, frame2_mz, frame2_height = merge_frames(
                    ms.deque_frame2, 3)

                # ms1: local max -- xic -- apex (防止frame重复) -- isotope
                idx_max1 = find_local_maximum(
                    frame1_at, frame1_mz, frame1_height,
                    tol_at_area=args.tol_im_area, tol_ppm=args.tol_ppm,
                    mz_min=args.pr_mz_min, at_min=0,
                    tol_point_num=9 # args.tol_point_num,
                )
                xics1 = get_xics(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, numba.typed.List(frame1_deque),
                    tol_at_area=args.tol_im_area, tol_ppm=args.tol_ppm,
                )
                # is_apex1 = (xics1[:, 3] > xics1[:, 2]) & (xics1[:, 3] > xics1[:, 4])
                is_apex1 = xics1.argmax(axis=-1) == 3
                idx_apex1 = idx_max1[is_apex1]

                # MS1cluster：M-1, M, M+1H
                # [n_max, charge range, isotope num]
                left_m, right_m, gaussian_m = find_isotope_cluster(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, is_apex1, xics1,
                    charge_min=args.charge_min, charge_max=args.charge_max,
                    tol_iso_num=args.tol_iso_num, tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_im_area, tol_at_shift=args.tol_im_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m, allow_lone=False)
                xics1 = xics1[is_apex1]
                cluster_idx = state_m.any(axis=-1)
                state_m = state_m[cluster_idx]
                xics1 = xics1[cluster_idx]
                idx_cluster1 = idx_apex1[cluster_idx]

                state_m_v.append(state_m)
            #     continue
            # state_m = np.vstack(state_m_v)
            # labels = state_m.sum(axis=0)
            # print(labels)
            # continue

                # ms2
                idx_max2 = find_local_maximum(
                    frame2_at, frame2_mz, frame2_height,
                    tol_at_area=args.tol_im_area, tol_ppm=args.tol_ppm,
                    tol_point_num=3, #args.tol_point_num,
                    mz_min=50, at_min=0.1,
                )
                xics2 = get_xics(
                    frame2_at, frame2_mz, frame2_height,
                    idx_max2, numba.typed.List(frame2_deque),
                    tol_at_area=args.tol_im_area, tol_ppm=args.tol_ppm,
                )

                # match
                pcc_ms2_m = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xics1, idx_max2, xics2,
                    tol_at_area=args.tol_im_area,
                    tol_at_shift=args.tol_im_shift,
                    tol_ppm=args.tol_ppm
                )

                # mgf
                cycle_rt = cycle_rts[cycle_i]
                for idx_col in np.arange(pcc_ms2_m.shape[1]):
                    pr_idx = idx_cluster1[idx_col]
                    pcc_v = pcc_ms2_m[:, idx_col]
                    # fg num
                    fg_num = (pcc_v > args.tol_pcc).sum()
                    if fg_num < args.tol_fg_num:
                        continue
                    # charge
                    pr_charges = np.where(state_m[idx_col])[0] + args.charge_min
                    # pr
                    pr_at = frame1_at[pr_idx]
                    pr_mz = frame1_mz[pr_idx]
                    pr_height = frame1_height[pr_idx]
                    # fg
                    fg_idx = idx_max2[pcc_v > args.tol_pcc]
                    scan_mz = frame2_mz[fg_idx]
                    scan_height = frame2_height[fg_idx]
                    scan_pcc = pcc_v[pcc_v > args.tol_pcc]
                    assert len(scan_mz) == len(scan_pcc)

                    # write
                    for pr_charge in pr_charges:
                        buffer_write("BEGIN IONS\n")
                        buffer_write(f"TITLE={pseudo_count}.{pr_charge}\n")
                        pseudo_count += 1
                        buffer_write(f"RTINSECONDS={cycle_rt:.2f}\n")
                        buffer_write(f"AT={pr_at:.2f}\n")
                        buffer_write(f"PEPMASS={pr_mz:.6f} {pr_height:.2f}\n")
                        buffer_write(f"CHARGE={pr_charge}+\n")
                        if scan_mz.size:
                            mz_str = np.char.mod("%.6f", scan_mz)
                            intensity_str = np.char.mod("%.2f", scan_height)
                            pcc_str = np.char.mod("%.2f", scan_pcc)
                            tmp = np.char.add(np.array(mz_str, dtype=str), ' ')
                            tmp = np.char.add(
                                tmp, np.array(intensity_str, dtype=str)
                            )
                            tmp = np.char.add(tmp, ' ')
                            if args.write_pcc:
                                tmp = np.char.add(
                                    tmp, np.array(pcc_str, dtype=str)
                                )
                            buffer_write("\n".join(tmp.tolist()))
                            buffer_write("\n")
                        buffer_write("END IONS\n\n")
                        if buffer.tell() >= MGF_BUFFER_FLUSH:
                            f.write(buffer.getvalue())
                            buffer.seek(0)
                            buffer.truncate(0)
            state_m = np.vstack(state_m_v)
            labels = state_m.sum(axis=0)
            logger.info(f'{quad_idx}, {labels}')
        remaining = buffer.getvalue()
        if remaining:
            f.write(remaining)