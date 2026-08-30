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
    assert (ats.max() > 100 and ats.max() < 450)
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
    n_seed = 0
    with open(outdir, "wb", buffering=1024*1024*50) as f:
        buffer = bytearray()
        counter = 0
        # cross-cycle consensus: (mz, at, charge) -> merged precursor record
        consensus = {}
        for frame_i in range(start, len(frame_rts) - start):
            frame_rt = frame_rts[frame_i]
            if frame_levels[frame_i] != 2: # level-1 --> MS2
                continue

            # load frames
            mbi.load_frames_to_deque(int(frame_i))
            frame1_deque = mbi.deque_frame1
            frame2_deque = mbi.deque_frame2

            # merge frames for maximum points
            frame1_at, frame1_mz, frame1_height = merge_frames(
                mbi.deque_frame1, 3, weighted=args.merge_weighted,
                at_tol=args.merge_at_tol)
            frame2_at, frame2_mz, frame2_height = merge_frames(
                mbi.deque_frame2, 3, weighted=args.merge_weighted,
                at_tol=args.merge_at_tol)
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
            frames1_list = None
            if mode in ['xic', 'xix']:
                frames1_list = numba.typed.List(frame1_deque)
                xics1 = get_xics(
                    frame1_at, frame1_mz, frame1_height,
                    idx_max1, frames1_list,
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
                    frames1_list,
                    charge_min=args.charge_min, charge_max=args.charge_max,
                    tol_iso_num=args.tol_iso_num,
                    iso_int_min=args.iso_int_min,
                    iso_int_max=args.iso_int_max,
                    tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc,
                    iso_rescue=args.iso_rescue,
                    iso_rescue_pcc=args.iso_rescue_pcc,
                    iso_rescue_gauss=args.iso_rescue_gauss
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m,
                                     allow_lone=args.allow_lone)
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
                    tol_iso_num=args.tol_iso_num,
                    iso_int_min=args.iso_int_min,
                    iso_int_max=args.iso_int_max,
                    tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m,
                                     allow_lone=args.allow_lone)
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
                    tol_iso_num=args.tol_iso_num,
                    iso_int_min=args.iso_int_min,
                    iso_int_max=args.iso_int_max,
                    tol_ppm=args.tol_ppm,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_pcc=args.tol_pcc
                )
                right_m = np.all(right_m, axis=-1)
                state_m = get_states(left_m, right_m, gaussian_m,
                                     allow_lone=args.allow_lone)
                xims1, xics1 = xims1[is_apex1], xics1[is_apex1]
                cluster_idx = state_m.any(axis=-1)
                state_m = state_m[cluster_idx]
                xims1, xics1 = xims1[cluster_idx], xics1[cluster_idx]
                idx_cluster1 = idx_apex1[cluster_idx]

            if len(idx_cluster1) == 0:
                continue

            # apex-only: emit spectra only near the apex frame of each
            # precursor XIC to remove cross-frame redundancy
            if args.apex_only and mode in ('xic', 'xix'):
                mid = across_cycle_num // 2
                center = xics1[:, mid]
                neighbors = np.maximum(xics1[:, mid - 1], xics1[:, mid + 1])
                keep = center >= 0.95 * neighbors
                idx_cluster1 = idx_cluster1[keep]
                state_m = state_m[keep]
                xics1 = xics1[keep]
                if mode == 'xix':
                    xims1 = xims1[keep]
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
                max2_ints = xics2[:, int(across_cycle_num/2)]
            elif mode == 'xim':
                pcc_ms2_m = find_frag_match(
                    frame1_at, frame1_mz, frame1_height,
                    frame2_at, frame2_mz, frame2_height,
                    idx_cluster1, xims1, idx_max2, xims2,
                    tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                    tol_ppm=args.tol_ppm
                )
                max2_ints = xims2.sum(axis=-1)
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
                max2_ints = xics2[:, int(across_cycle_num/2)]

            print_log(frame_i, len(frame_rts),
                      frame1_at, idx_max1, idx_apex1, idx_cluster1,
                      frame2_at, idx_max2
                      )
            n_seed += state_m.sum()

            # mgf
            for idx_col in np.arange(pcc_ms2_m.shape[1]):
                pr_idx = idx_cluster1[idx_col]
                pcc_v = pcc_ms2_m[:, idx_col]
                # fg num
                pcc_good = pcc_v > args.tol_pcc
                fg_num = pcc_good.sum()
                if fg_num < args.tol_fg_num:
                    continue
                # strong-match quality gate: spectra must contain enough
                # well-correlated fragments, otherwise they dilute FDR
                if args.tol_fg_num_strong > 0:
                    fg_num_strong = (pcc_v > args.tol_pcc_strong).sum()
                    if fg_num_strong < args.tol_fg_num_strong:
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
                if args.frag_ppm_shift:
                    scan_mz = scan_mz * (1 + args.frag_ppm_shift * 1e-6)
                if args.xic_mid_int:
                    scan_height = max2_ints[pcc_good]
                else:
                    scan_height = frame2_height[fg_idx]
                assert len(scan_mz) == len(scan_height)

                if args.consensus:
                    # merge into cross-cycle consensus instead of per-cycle write
                    for pr_charge in pr_charges:
                        charge_int = int(pr_charge)
                        key = (round(float(pr_mz), 2), round(float(pr_at), 1),
                               charge_int)
                        ent = consensus.get(key)
                        if ent is None:
                            ent = {'mz': float(pr_mz), 'at': float(pr_at),
                                   'height': float(pr_height),
                                   'rt': float(frame_rt), 'frags': {}}
                            consensus[key] = ent
                        if pr_height > ent['height']:
                            ent['mz'] = float(pr_mz)
                            ent['at'] = float(pr_at)
                            ent['height'] = float(pr_height)
                            ent['rt'] = float(frame_rt)
                        frags = ent['frags']
                        for m, h in zip(scan_mz, scan_height):
                            b = round(float(m), 3)
                            cur = frags.get(b)
                            if cur is None or h > cur[1]:
                                frags[b] = (float(m), float(h))
                    continue

                # 不同charge也是相同scan_mz
                if args.write_pcc:
                    scan_pcc = pcc_v[pcc_good]
                    peak_str = ("\n".join(
                        f"{m:.6f} {h:.2f} {p:.2f}" for m, h, p in
                        zip(scan_mz, scan_height, scan_pcc)) + "\n").encode()
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
        if args.consensus:
            for (mz_k, at_k, charge_int), ent in consensus.items():
                if len(ent['frags']) < args.tol_fg_num:
                    continue
                peaks = sorted(ent['frags'].values())
                scan_mz_c = np.array([p[0] for p in peaks], dtype=np.float32)
                scan_h_c = np.array([p[1] for p in peaks], dtype=np.float32)
                peak_block = format_mz_int(scan_mz_c, scan_h_c) + b"END IONS\n\n"
                common_header = (
                    f"RTINSECONDS={ent['rt']:.2f}\n"
                    f"AT={ent['at']:.2f}\n"
                    f"PEPMASS={ent['mz']:.6f} {ent['height']:.2f}\n").encode()
                counter += 1
                buffer.extend(
                    f"BEGIN IONS\nTITLE={counter}.{charge_int}\n".encode())
                buffer.extend(common_header)
                buffer.extend(f"CHARGE={charge_int}+\n".encode())
                buffer.extend(peak_block)
                if len(buffer) >= MGF_BUFFER_FLUSH:
                    f.write(buffer)
                    buffer.clear()
        if buffer:
            f.write(buffer)
        logger.info(f'n_seed: {n_seed}, n_spectra: {counter}')
