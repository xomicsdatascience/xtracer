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
    start = across_cycle_num

    with open(outdir, "w") as f:
        buffer = StringIO()
        buffer_write = buffer.write
        pseudo_count = 0
        quad_num = ms.get_quad_num()
        for quad_idx in range(1, 1+quad_num):
            # loop cycle
            frame_rts = ms.get_frame_times(quad_idx)
            frame_levels = ms.get_frame_levels(quad_idx)

            xics1_v, xics2_v = [], []
            ats1_v, mzs1_v, ints1_v, cycle1_v = [], [], [], []
            ats2_v, mzs2_v, ints2_v, cycle2_v = [], [], [], []
            for frame_i in range(start, len(frame_rts) - start):
                if frame_levels[frame_i] != 1:
                    continue

                # load frames
                ms.load_frames_to_deque(quad_idx, frame_i)
                frame1_deque = ms.deque_frame1
                frame2_deque = ms.deque_frame2

                # merge frames for maximum points
                frame1_at, frame1_mz, frame1_height = merge_frames(
                    ms.deque_frame1, 3)
                frame2_at, frame2_mz, frame2_height = merge_frames(
                    ms.deque_frame2, 3)

                # local point -- xic -- apex
                for (ms_level, frame_at, frame_mz, frame_height, frame_deque) in (
                    [1, frame1_at, frame1_mz, frame1_height, frame1_deque],
                    [2, frame2_at, frame2_mz, frame2_height, frame2_deque],
                ):
                    idx_max_points = find_local_maximum(
                        frame_at, frame_mz, frame_height,
                        tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                        mz_min=args.pr_mz_min, at_min=args.at_min,
                        tol_point_num=args.tol_point_num,
                    )
                    xics = get_xics(
                        frame_at, frame_mz, frame_height,
                        idx_max_points, numba.typed.List(frame_deque),
                        tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                    )
                    condition1 = (xics[:, 3] >= xics[:, 4]) & (xics[:, 3] >= xics[:, 2])
                    argmax = xics.argmax(axis=-1)
                    condition2 = (argmax >= 2) & (argmax <= 4)
                    good_idx = condition1 & condition2
                    idx_max_points = idx_max_points[good_idx]
                    if ms_level == 1:
                        xics1_v.append(xics[good_idx])
                        ats1_v.append(frame_at[idx_max_points])
                        mzs1_v.append(frame_mz[idx_max_points])
                        ints1_v.append(frame_height[idx_max_points])
                        cycle1_v.append([frame_i//2] * len(idx_max_points))
                    else:
                        xics2_v.append(xics[good_idx])
                        ats2_v.append(frame_at[idx_max_points])
                        mzs2_v.append(frame_mz[idx_max_points])
                        ints2_v.append(frame_height[idx_max_points])
                        cycle2_v.append([frame_i//2] * len(idx_max_points))
            cycles1, ats1, mzs1, ints1, xics1 = get_points(
                xics1_v, ats1_v, mzs1_v, ints1_v, cycle1_v
            )
            cycles2, ats2, mzs2, ints2, xics2 = get_points(
                xics2_v, ats2_v, mzs2_v, ints2_v, cycle2_v
            )

            # 同位汇聚 for ms1
            labels = assign_mono_labels(
                mzs1, ats1, cycles1, ints1, args.tol_ppm, 2, 3
            )
            print({int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})
            continue
    #         idx = labels >= 1
    #         mzs1, ats1, cycles1, ints1, xics1 = (
    #             mzs1[idx], ats1[idx], cycles1[idx], ints1[idx], xics1[idx]
    #         )
    #         charges1 = labels[idx]
    #
    #         if len(mzs1) == 0:
    #             continue
    #
    #         # match
    #         n_prec = len(mzs1)
    #         n_frag = len(mzs2)
    #         count = 0
    #         for i in range(n_prec):
    #             prec_mz = mzs1[i]
    #             prec_at = ats1[i]
    #             prec_cycle = cycles1[i]
    #             prec_int = ints1[i]
    #             prec_charge = charges1[i]
    #             prec_xic = xics1[i]
    #
    #             if np.std(prec_xic) < 1e-10:
    #                 continue
    #
    #             cycle_mask = np.abs(cycles2 - prec_cycle) <= 3  # Cycle 容忍度
    #             at_mask = np.abs(ats2 - prec_at) <= args.tol_at_shift
    #             candidate_mask = cycle_mask & at_mask
    #             candidate_idx = np.where(candidate_mask)[0]
    #
    #             if len(candidate_idx) < args.tol_fg_num:
    #                 continue
    #
    #             pcc_values = np.zeros(len(candidate_idx))
    #             for j, idx in enumerate(candidate_idx):
    #                 frag_xic = xics2[idx]
    #                 if np.std(frag_xic) < 1e-10:
    #                     pcc_values[j] = 0
    #                 else:
    #                     pcc_values[j] = np.corrcoef(prec_xic, frag_xic)[0, 1]
    #
    #             good_mask = pcc_values >= args.tol_pcc
    #             good_local_idx = candidate_idx[good_mask]
    #             good_pcc = pcc_values[good_mask]
    #
    #             if len(good_local_idx) < args.tol_fg_num:
    #                 continue
    #
    #             prec_rt = frame_rts[int(prec_cycle)]
    #
    #             buffer_write("BEGIN IONS\n")
    #             buffer_write(f"TITLE={pseudo_count}.{prec_charge}\n")
    #             buffer_write(f"RTINSECONDS={prec_rt:.2f}\n")
    #             buffer_write(f"PEPMASS={prec_mz:.6f} {prec_int:.2f}\n")
    #             buffer_write(f"CHARGE={prec_charge}+\n")
    #             buffer_write(f"AT={prec_at:.4f}\n")
    #
    #             # 写入片段离子
    #             for frag_idx, pcc in zip(good_local_idx, good_pcc):
    #                 mz = mzs2[frag_idx]
    #                 intensity = ints2[frag_idx]
    #                 if args.write_pcc:
    #                     buffer_write(f"{mz:.6f} {intensity:.2f} {pcc:.2f}\n")
    #                 else:
    #                     buffer_write(f"{mz:.6f} {intensity:.2f}\n")
    #
    #             buffer_write("END IONS\n\n")
    #             count += 1
    #             pseudo_count += 1
    #             # 缓冲区刷新
    #             if buffer.tell() >= MGF_BUFFER_FLUSH:
    #                 f.write(buffer.getvalue())
    #                 buffer.seek(0)
    #                 buffer.truncate(0)
    #
    #         logger.info(
    #             f"Quad {quad_idx}: {n_prec} precursors → {count} pseudo-spectra")
    #
    #         # 写入剩余缓冲区
    #     remaining = buffer.getvalue()
    #     if remaining:
    #         f.write(remaining)
    #
    # logger.info(f"Total pseudo-spectra written: {pseudo_count}")
