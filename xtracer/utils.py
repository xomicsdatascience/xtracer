import numpy as np
from numba import jit, prange
from xtracer.log import Logger

logger = Logger.get_logger()

try:
    profile
except:
    profile = lambda x: x


C13_DELTA = 1.0033548378
MGF_BUFFER_FLUSH = 5_000_000  # ~5 MB


@jit(nopython=True)
def smooth_vec(vec):
    # 平滑
    vec_smooth = np.empty_like(vec)
    vec_smooth[0] = 0.667 * vec[0] + 0.333 * vec[1]
    vec_smooth[-1] = 0.667 * vec[-1] + 0.333 * vec[-2]
    for i in range(1, len(vec) - 1):
        vec_smooth[i] = 0.25 * vec[i - 1] + 0.5 * vec[i] + 0.25 * vec[i + 1]
    return vec_smooth


@jit(nopython=True)
def cal_pcc(vec1, vec2):
    n = len(vec1)
    mean1 = 0.0
    mean2 = 0.0
    for i in range(n):
        mean1 += vec1[i]
        mean2 += vec2[i]
    mean1 /= n
    mean2 /= n

    num = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for i in range(n):
        diff1 = vec1[i] - mean1
        diff2 = vec2[i] - mean2
        num += diff1 * diff2
        denom1 += diff1 * diff1
        denom2 += diff2 * diff2

    if denom1 < 1e-12 or denom2 < 1e-12:
        return 0.0
    return num / np.sqrt(denom1 * denom2)


@jit(nopython=True)
def merge_frames_core(at1, mz1, h1, at2, mz2, h2, mz_tol=0.0001, at_tol=0.001):
    n1 = len(at1)
    n2 = len(at2)

    merged_at = np.empty(n1 + n2, dtype=np.float32)
    merged_mz = np.empty(n1 + n2, dtype=np.float32)
    merged_h = np.empty(n1 + n2, dtype=np.float32)

    i = j = idx = 0

    while i < n1 and j < n2:
        mz_diff = mz1[i] - mz2[j]
        at_diff = abs(at1[i] - at2[j])

        if abs(mz_diff) <= mz_tol and at_diff <= at_tol:
            merged_mz[idx] = (mz1[i] + mz2[j]) * 0.5
            merged_at[idx] = (at1[i] + at2[j]) * 0.5
            merged_h[idx] = h1[i] + h2[j]
            idx += 1
            i += 1
            j += 1
        elif mz1[i] < mz2[j] - mz_tol:
            merged_mz[idx] = mz1[i]
            merged_at[idx] = at1[i]
            merged_h[idx] = h1[i]
            idx += 1
            i += 1
        else:
            merged_mz[idx] = mz2[j]
            merged_at[idx] = at2[j]
            merged_h[idx] = h2[j]
            idx += 1
            j += 1

    # rest points
    while i < n1:
        merged_mz[idx] = mz1[i]
        merged_at[idx] = at1[i]
        merged_h[idx] = h1[i]
        idx += 1
        i += 1
    while j < n2:
        merged_mz[idx] = mz2[j]
        merged_at[idx] = at2[j]
        merged_h[idx] = h2[j]
        idx += 1
        j += 1

    return merged_at[:idx], merged_mz[:idx], merged_h[:idx]


@profile
def merge_frames(deque_frame, merge_num=None):
    if merge_num is None:
        merge_num = len(deque_frame)

    assert len(deque_frame) >= merge_num
    assert merge_num % 2 == 1

    frame_num = len(deque_frame)
    center_idx = frame_num // 2
    half = merge_num // 2
    start = max(0, center_idx - half)
    end = min(frame_num, center_idx + half + 1)

    frame_at, frame_mz, frame_h = deque_frame[start]
    for frame_i in range(start + 1, end):
        at, mz, h = deque_frame[frame_i]
        frame_at, frame_mz, frame_h = merge_frames_core(
            frame_at, frame_mz, frame_h, at, mz, h
        )
    assert np.all(frame_mz[:-1] <= frame_mz[1:])  # sorted
    return frame_at, frame_mz, frame_h


@jit(nopython=True, parallel=True)
def find_local_maximum(
        frame_at, frame_mz, frame_height,
        tol_at_area, tol_ppm, tol_point_num, mz_min, at_min
):
    result_is_max = np.zeros(len(frame_at), dtype=np.bool_)

    # sum
    for i in prange(len(frame_at)):
        i_mz = frame_mz[i]
        i_at = frame_at[i]
        if (i_mz < mz_min) or (i_at < at_min):
            continue
        i_height = frame_height[i]
        is_max = True
        area_cnt = 1
        # check right
        for ii in range(i + 1, len(frame_at)):
            ii_at = frame_at[ii]
            ii_mz = frame_mz[ii]
            ii_height = frame_height[ii]
            bias_ppm = 1e6 * abs(ii_mz - i_mz) / i_mz
            bias_at = abs(ii_at - i_at)
            if bias_ppm > tol_ppm:
                break
            if bias_at > tol_at_area:
                continue
            area_cnt += 1
            if ii_height > i_height:
                is_max = False
        # check left
        for ii in range(i - 1, -1, -1):
            ii_at = frame_at[ii]
            ii_mz = frame_mz[ii]
            ii_height = frame_height[ii]
            bias_ppm = 1e6 * abs(ii_mz - i_mz) / i_mz
            bias_at = abs(ii_at - i_at)
            if bias_ppm > tol_ppm:
                break
            if bias_at > tol_at_area:
                continue
            area_cnt += 1
            if ii_height > i_height:
                is_max = False
        if is_max and (area_cnt > tol_point_num):
            result_is_max[i] = is_max
    return np.where(result_is_max == True)[0]


@jit(nopython=True)
def get_xim(frame_at, frame_mz, frame_height,
            idx, tol_at_area, tol_ppm, n_bins
    ):
    mz_ref = frame_mz[idx]
    im_center = frame_at[idx]
    vec = np.zeros(n_bins, dtype=np.float32)
    bin_width = 2 * tol_at_area / n_bins
    im_start = im_center - tol_at_area

    # to right
    for ii in range(idx, len(frame_at)):
        ppm = 1e6 * (frame_mz[ii] - mz_ref) / mz_ref
        if ppm > tol_ppm:
            break  # m/z overflow
        if abs(frame_at[ii] - im_center) > tol_at_area:
            continue
        bin_idx = int((frame_at[ii] - im_start) / bin_width)
        if 0 <= bin_idx < n_bins:
            vec[bin_idx] += frame_height[ii]

    # to left
    for ii in range(idx - 1, -1, -1):
        ppm = 1e6 * (frame_mz[ii] - mz_ref) / mz_ref
        if abs(ppm) > tol_ppm:
            break
        if abs(frame_at[ii] - im_center) > tol_at_area:
            continue
        bin_idx = int((frame_at[ii] - im_start) / bin_width)
        if 0 <= bin_idx < n_bins:
            vec[bin_idx] += frame_height[ii]
    vec = smooth_vec(vec)
    return vec


@jit(nopython=True, parallel=True)
def get_xims(frame_at, frame_mz, frame_height,
             idx_max_points,
             tol_at_area, tol_ppm, n_bins=15):
    xims = np.zeros((len(idx_max_points), n_bins), dtype=np.float32)
    for idx_i in prange(len(idx_max_points)):
        i = idx_max_points[idx_i]
        xim = get_xim(
            frame_at, frame_mz, frame_height, i,
            tol_at_area=tol_at_area, tol_ppm=tol_ppm,
            n_bins=n_bins
        )
        xims[idx_i, :] = xim
    return xims


@jit(nopython=True)
def get_xic(frames, target_at, target_mz,
            tol_at_area, tol_ppm
            ):
    mz_limit_left = target_mz * (1 - tol_ppm * 1e-6)
    mz_limit_right = target_mz * (1 + tol_ppm * 1e-6)
    at_limit_left = target_at - tol_at_area
    at_limit_right = target_at + tol_at_area

    vec = np.zeros(len(frames), dtype=np.float32)

    for frame_i in range(len(frames)):
        frame_at, frame_mz, frame_height = frames[frame_i]
        for i in range(len(frame_at)):
            at, mz = frame_at[i], frame_mz[i]
            if mz < mz_limit_left:
                continue
            if mz > mz_limit_right:
                break
            if at < at_limit_left:
                continue
            if at > at_limit_right:
                continue
            vec[frame_i] += frame_height[i]
    vec = smooth_vec(vec)
    return vec


@jit(nopython=True, parallel=True)
def get_xics(frame_at, frame_mz, frame_height,
             idx_max_points, frames,
             tol_at_area, tol_ppm):
    xics = np.zeros((len(idx_max_points), len(frames)), dtype=np.float32)
    for idx_i in prange(len(idx_max_points)):
        i = idx_max_points[idx_i]
        target_at = frame_at[i]
        target_mz = frame_mz[i]
        xic = get_xic(
            frames, target_at, target_mz,
            tol_at_area=tol_at_area, tol_ppm=tol_ppm,
        )
        xics[idx_i, :] = xic
    return xics


@jit(nopython=True, parallel=True)
def find_isotope_cluster(
        frame1_at, frame1_mz, frame1_height,
        idx_max_points, xixs1,
        charge_min, charge_max, tol_iso_num,
        tol_at_area, tol_at_shift, tol_ppm
):
    results = np.zeros((len(idx_max_points),
                        (charge_max - charge_min + 1),
                        tol_iso_num))

    for idx_i in prange(len(idx_max_points)):
        i = idx_max_points[idx_i]
        i_at = frame1_at[i]
        i_mz = frame1_mz[i]
        limit_mz = (i_mz + tol_iso_num * C13_DELTA) * (1 + 100 * 1e-6)
        vec1 = xixs1[idx_i]
        for idx_ii in range(idx_i + 1, len(idx_max_points)): # local maximum
            ii = idx_max_points[idx_ii]
            ii_at = frame1_at[ii]
            bias_at = abs(ii_at - i_at)
            if bias_at > tol_at_shift:
                continue
            ii_mz = frame1_mz[ii]
            if ii_mz > limit_mz:
                break
            for n_charge in range(charge_min, charge_max+1):
                for n_neutron in range(1, tol_iso_num+1):
                    target_mz = i_mz + n_neutron * C13_DELTA / n_charge
                    bias_ppm = 1e6 * abs(ii_mz - target_mz) / target_mz
                    if bias_ppm > tol_ppm:
                        continue
                    vec2 = xixs1[idx_ii]
                    pcc = cal_pcc(vec1, vec2)
                    # maximum pcc in case multiple points
                    cur = results[idx_i, n_charge-charge_min, n_neutron-1]
                    pcc = max(pcc, cur)
                    results[idx_i, n_charge-charge_min, n_neutron-1] = pcc
    return results


@jit(nopython=True, parallel=True)
def find_frag_match(
        frame1_at, frame1_mz, frame1_height,
        frame2_at, frame2_mz, frame2_height,
        idx_cluster1_points, xims1, idx_max2_points, xims2,
        tol_at_area, tol_at_shift, tol_ppm
):
    # record the pcc between frag ions to precursors
    results_pcc = -np.ones(
        (len(idx_max2_points), len(idx_cluster1_points)),
        dtype=np.float32
    )

    for idx_fg in prange(len(idx_max2_points)):
        i_fg = idx_max2_points[idx_fg]
        fg_at = frame2_at[i_fg]
        fg_xim = xims2[idx_fg]
        for idx_pr in range(len(idx_cluster1_points)):
            i_pr = idx_cluster1_points[idx_pr]
            pr_at = frame1_at[i_pr]
            bias_at = abs(fg_at - pr_at)
            if bias_at > tol_at_shift:
                continue
            pr_xim = xims1[idx_pr]
            pcc = cal_pcc(pr_xim, fg_xim)
            results_pcc[idx_fg, idx_pr] = pcc
    return results_pcc


def plot_area_scatter(frame_mz, frame_at, frame_y, idx_center):
    mz, at = frame_mz[idx_center], frame_at[idx_center]
    mz_min = mz * (1 - 50*1e-6)
    mz_max = mz * (1 + 50*1e-6)
    at_min = at - 2.5
    at_max = at + 2.5
    idx = ((frame_at < at_max) &
           (frame_at > at_min) &
           (frame_mz < mz_max) &
           (frame_mz > mz_min))
    mzs, ats, ys = frame_mz[idx], frame_at[idx], frame_y[idx]
    plt.scatter(mzs, ats, c=ys, marker='s', s=300)
    plt.xlim((mz_min, mz_max))
    plt.ylim((at_min, at_max))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.grid(True)
    plt.show()


def compute_log_points(n_total, n_print=10):
    units = [10000, 1000, 100, 50]
    candidates = set()
    for unit in units:
        for i in range(unit, n_total, unit):
            candidates.add(i)

    candidates = sorted(candidates)

    if len(candidates) <= n_print:
        return candidates

    step = len(candidates) / n_print
    log_points = [candidates[int(round(i * step))] for i in range(n_print)]
    return np.array(log_points)


def print_log(
        frame_i, n_total,
        frame1_at, idx_max1_points, idx_cluster1_points,
        frame2_at, idx_max2_points
):
    log_points = compute_log_points(n_total, n_print=10)
    if frame_i in log_points:
        info = (f'{frame_i}/{n_total}, '
                f'{len(frame1_at)}, '
                f'{len(idx_max1_points)}, '
                f'{len(idx_cluster1_points)}, '
                f'{len(frame2_at)}, '
                f'{len(idx_max2_points)}'
        )
        logger.info(info)
    if frame_i+1 in log_points:
        info = (f'{frame_i}/{n_total}, '
                f'{len(frame1_at)}, '
                f'{len(idx_max1_points)}, '
                f'{len(idx_cluster1_points)}, '
                f'{len(frame2_at)}, '
                f'{len(idx_max2_points)}'
        )
        logger.info(info)