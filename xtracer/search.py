import numpy as np

import numba.typed
from xtracer.utils import *
from xtracer.mbi import MBIReader
from io import StringIO

try:
    profile
except:
    profile = lambda x: x


def save_frame_result(frame_rt, frame_ats, frame_mzs, idx):
    rts = [frame_rt] * len(idx)
    ats = frame_ats[idx]
    mzs = frame_mzs[idx]
    return rts, ats, mzs


def check_ms(ats, mzs, ints):
    assert (ats.max() > 300 and ats.max() < 450)
    assert (mzs.min() > 10 and mzs.max() < 5000)
    assert ints.min() > 0


@profile
def main(args, indir, outdir, mode):
    # read .mbi
    if mode == 'xim':
        across_cycle_num = args.xim_across_cycle_num
    else:
        across_cycle_num = args.xic_across_cycle_num

    mbi = MBIReader(indir, across_cycle_num)
    start = across_cycle_num

    # loop cycle
    frame_rts = np.array(mbi.GetRetentionTimes())
    frame_levels = np.array(mbi.GetFrameMSLevels())
    tmp_max, tmp_apex, tmp_cluster = [], [], []
    with open(outdir, "wb", buffering=1024*1024*50) as f:
        buffer = bytearray()
        counter = 0
        for frame_i in range(start, len(frame_rts) - start):
            frame_rt = frame_rts[frame_i]
            if frame_levels[frame_i] != 2: # level-1 --> MS2
                continue

            # load frames
            mbi.load_frames_to_deque(int(frame_i))
            frame1_deque = mbi.deque_frame1
            frame2_deque = mbi.deque_frame2

            # merge frames for maximum points
            frame1_at, frame1_mz, frame1_height = merge_frames(mbi.deque_frame1, 3)
            frame2_at, frame2_mz, frame2_height = merge_frames(mbi.deque_frame2, 3)
            check_ms(frame1_at, frame1_mz, frame1_height)
            check_ms(frame2_at, frame2_mz, frame2_height)

            # local maximum points
            idx_max1 = find_local_maximum(
                frame1_at, frame1_mz, frame1_height,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                mz_min=args.pr_mz_min, at_min=args.at_min,
                tol_point_num=args.tol_neighbor1_num,
            )

            # extract
            xics1, xims1 = None, None
            if mode in ['xic', 'xix']:
                xics1 = get_xics(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, numba.typed.List(frame1_deque),
                    tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                )
            if mode in ['xim', 'xix']:
                xims1 = get_xims(
                    frame1_at, frame1_mz, frame1_height, idx_max1,
                    tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                )

            # MS1cluster：M, M+1H, M+2H
            # [n_max, charge range, isotope num]
            if mode == 'xic':
                # is_apex1 = (xics1[:, 3] >= xics1[:, 4]) & (xics1[:, 3] >= xics1[:, 2])
                is_apex1 = xics1[:, 3] > 0
                idx_apex1 = idx_max1[is_apex1]
                left_m, right_m, gaussian_m = find_isotope_cluster_xic(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, is_apex1, xics1,
                    charge_min=args.charge_min, charge_max=args.charge_max,
                    tol_iso_num=args.tol_iso_num, tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m, allow_lone=False)
                xics1 = xics1[is_apex1]
                cluster_idx = state_m.any(axis=-1)
                state_m = state_m[cluster_idx]
                xics1 = xics1[cluster_idx]
                idx_cluster1 = idx_apex1[cluster_idx]

                # tmp_max.append(save_frame_result(frame_rt, frame1_at, frame1_mz, idx_max1))
                # tmp_apex.append(save_frame_result(frame_rt, frame1_at, frame1_mz, idx_apex1))
                # tmp_cluster.append(save_frame_result(frame_rt, frame1_at, frame1_mz, idx_cluster1))
                # continue
        # cal_recall(tmp_max)
        # cal_recall(tmp_apex)
        # cal_recall(tmp_cluster)
            if mode == 'xim':
                is_apex1 = np.ones(len(idx_max1), dtype=bool)
                idx_apex1 = idx_max1[is_apex1]
                left_m, right_m, gaussian_m = find_isotope_cluster_xim(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, is_apex1, xims1,
                    charge_min=args.charge_min, charge_max=args.charge_max,
                    tol_iso_num=args.tol_iso_num, tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m, allow_lone=False)
                xims1 = xims1[is_apex1]
                cluster_idx = state_m.any(axis=-1)
                state_m = state_m[cluster_idx]
                xims1 = xims1[cluster_idx]
                idx_cluster1 = idx_apex1[cluster_idx]
            if mode == 'xix':
                is_apex1 = xics1[:, 3] > 0
                idx_apex1 = idx_max1[is_apex1]
                left_m, right_m, gaussian_m = find_isotope_cluster_xix(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, is_apex1, xics1, xims1,
                    charge_min=args.charge_min, charge_max=args.charge_max,
                    tol_iso_num=args.tol_iso_num, tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m, allow_lone=False)
                xims1, xics1 = xims1[is_apex1], xics1[is_apex1]
                cluster_idx = state_m.any(axis=-1)
                state_m = state_m[cluster_idx]
                xims1, xics1 = xims1[cluster_idx], xics1[cluster_idx]
                idx_cluster1 = idx_apex1[cluster_idx]

            if len(idx_cluster1) == 0:
                continue

            # ms2
            idx_max2 = find_local_maximum(
                frame2_at, frame2_mz, frame2_height,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                tol_point_num=args.tol_neighbor2_num,
                mz_min=100, at_min=args.at_min,
            )
            if mode in ['xic', 'xix']:
                xics2 = get_xics(
                    frame2_at, frame2_mz, frame2_height,
                    idx_max2, numba.typed.List(frame2_deque),
                    tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                )
            if mode in ['xim', 'xix']:
                xims2 = get_xims(
                    frame2_at, frame2_mz, frame2_height, idx_max2,
                    tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                )

            # match
            if mode == 'xic':
                pcc_ms2_m = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xics1, idx_max2, xics2,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_ppm=args.tol_ppm
                )
            elif mode == 'xim':
                pcc_ms2_m = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xims1, idx_max2, xims2,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_ppm=args.tol_ppm
                )
            elif mode == 'xix':
                pcc_ms2_m_xic = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xics1, idx_max2, xics2,
                    tol_at_area=args.tol_at_area,
                    tol_at_shift=args.tol_at_shift,
                    tol_ppm=args.tol_ppm
                )
                pcc_ms2_m_xim = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xims1, idx_max2, xims2,
                    tol_at_area=args.tol_at_area,
                    tol_at_shift=args.tol_at_shift,
                    tol_ppm=args.tol_ppm
                )
                pcc_ms2_m = (pcc_ms2_m_xic + pcc_ms2_m_xim) / 2

            print_log(frame_i, len(frame_rts),
                      frame1_at, idx_max1, idx_apex1, idx_cluster1,
                      frame2_at, idx_max2
                      )

            # mgf
            for idx_col in np.arange(pcc_ms2_m.shape[1]):
                pr_idx = idx_cluster1[idx_col]
                pcc_v = pcc_ms2_m[:, idx_col]
                # fg num
                pcc_good = pcc_v > args.tol_pcc
                fg_num = pcc_good.sum()
                if fg_num < args.tol_fg_num:
                    continue
                # charge
                pr_charges = np.where(state_m[idx_col])[0] + args.charge_min
                # pr
                pr_at = frame1_at[pr_idx]
                pr_mz = frame1_mz[pr_idx]
                pr_height = frame1_height[pr_idx]
                # fg
                fg_idx = idx_max2[pcc_good]
                scan_mz = frame2_mz[fg_idx]
                scan_height = frame2_height[fg_idx]
                assert len(scan_mz) == len(scan_height)
                # 不同charge也是相同scan_mz
                if args.write_pcc:
                    scan_pcc = pcc_v[pcc_good]
                    peak_str = "\n".join(
                        f"{m:.6f} {h:.2f} {p:.2f}" for m, h, p in
                        zip(scan_mz, scan_height, scan_pcc))
                else:
                    # peak_str = "\n".join([f"{m:.6f} {h:.2f}" for m, h in zip(scan_mz, scan_height)])
                    peak_str = format_mz_int(scan_mz, scan_height)
                peak_block = peak_str + b"END IONS\n\n"
                common_header = f"RTINSECONDS={frame_rt:.2f}\nAT={pr_at:.2f}\nPEPMASS={pr_mz:.6f} {pr_height:.2f}\n".encode()

                # write
                for pr_charge in pr_charges:
                    counter += 1
                    charge_int = int(pr_charge)
                    buffer.extend(f"BEGIN IONS\nTITLE={counter}.{charge_int}\n".encode())
                    buffer.extend(common_header)
                    buffer.extend(f"CHARGE={charge_int}+\n".encode())
                    buffer.extend(peak_block)
                    if len(buffer) >= MGF_BUFFER_FLUSH:
                        f.write(buffer)
                        buffer.clear()
        if buffer:
            f.write(buffer)

