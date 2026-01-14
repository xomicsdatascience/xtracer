from io import StringIO
from xtracer.mbi import MBIReader
from xtracer.utils import *

try:
    profile
except:
    profile = lambda x: x

@profile
def main(args, indir, outdir):
    # read .mbi
    mbi = MBIReader(indir, args.xim_across_cycle_num)

    # loop cycle
    frame_rts = np.array(mbi.GetRetentionTimes())
    frame_levels = np.array(mbi.GetFrameMSLevels())
    with open(outdir, "w") as f:
        buffer = StringIO()
        buffer_write = buffer.write
        counter = 1
        start = args.xim_across_cycle_num
        for frame_i in range(start, len(frame_rts) - start):
            if frame_levels[frame_i] != 2: # level-1 --> MS2
                continue

            # load frames
            mbi.load_frames_to_deque(frame_i)

            # merge frames
            frame1_at, frame1_mz, frame1_height = merge_frames(mbi.deque_frame1)
            frame2_at, frame2_mz, frame2_height = merge_frames(mbi.deque_frame2)

            # local maximum points with its xim
            idx_max1_points = find_local_maximum(
                frame1_at, frame1_mz, frame1_height,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                mz_min=args.pr_mz_min, at_min=args.at_min,
                tol_point_num=args.tol_point_num,
            )
            xims1 = get_xims(
                frame1_at, frame1_mz, frame1_height, idx_max1_points,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
            )

            # MS1cluster：M, M+1H, M+2H
            # [n_max, charge range, isotope num]
            pcc_cluster1_m = find_isotope_cluster(
                frame1_at, frame1_mz, frame1_height,
                idx_max1_points, xims1,
                charge_min=args.charge_min, charge_max=args.charge_max,
                tol_iso_num=args.tol_iso_num, tol_ppm=args.tol_ppm,
                tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
            )
            idx_tmp = np.any(
                np.all(pcc_cluster1_m > args.tol_pcc, axis=2),
                axis=1
            )
            idx_cluster1_points = idx_max1_points[idx_tmp]
            pcc_cluster1_m = pcc_cluster1_m[idx_tmp]
            xims1 = xims1[idx_tmp]
            if len(pcc_cluster1_m) == 0:
                continue

            # ms2
            idx_max2_points = find_local_maximum(
                frame2_at, frame2_mz, frame2_height,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
                tol_point_num=args.tol_point_num,
                mz_min=50, at_min=args.at_min,
            )
            xims2 = get_xims(
                frame2_at, frame2_mz, frame2_height, idx_max2_points,
                tol_at_area=args.tol_at_area, tol_ppm=args.tol_ppm,
            )

            # match each ms2 to clusters: [n_ms2, n_cluster1]
            pcc_ms2_m = find_frag_match(
                frame1_at, frame1_mz, frame1_height,
                frame2_at, frame2_mz, frame2_height,
                idx_cluster1_points, xims1, idx_max2_points, xims2,
                tol_at_area=args.tol_at_area, tol_at_shift=args.tol_at_shift,
                tol_ppm=args.tol_ppm
            )
            print_log(frame_i, len(frame_rts),
                      frame1_at, idx_max1_points, idx_cluster1_points,
                      frame2_at, idx_max2_points
                      )

            # mgf
            frame_rt = frame_rts[frame_i]
            for idx_col in np.arange(pcc_ms2_m.shape[1]):
                pr_idx = idx_cluster1_points[idx_col]
                pcc_v = pcc_ms2_m[:, idx_col]
                # fg num
                fg_num = (pcc_v > args.tol_pcc).sum()
                if fg_num < args.tol_fg_num:
                    continue
                # charge
                pcc_m = pcc_cluster1_m[idx_col]
                pr_charges = np.arange(args.charge_min,
                                       args.charge_max + 1)
                idx_charges = np.all(pcc_m > args.tol_pcc, axis=-1)
                pr_charges = pr_charges[idx_charges]
                if pr_charges.size == 0:
                    continue
                # pr
                pr_at = frame1_at[pr_idx]
                pr_mz = frame1_mz[pr_idx]
                pr_height = frame1_height[pr_idx]
                # fg
                fg_idx = idx_max2_points[pcc_v > args.tol_pcc]
                scan_mz = frame2_mz[fg_idx]
                scan_height = frame2_height[fg_idx]
                scan_pcc = pcc_v[pcc_v > args.tol_pcc]
                assert len(scan_mz) == len(scan_pcc)

                # write
                for pr_charge in pr_charges:
                    buffer_write("BEGIN IONS\n")
                    buffer_write(f"TITLE={counter}.{pr_charge}\n")
                    counter += 1
                    buffer_write(f"RTINSECONDS={frame_rt:.2f}\n")
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
        remaining = buffer.getvalue()
        if remaining:
            f.write(remaining)



if __name__ == "__main__":
    main()
