import numpy as np
from numba import jit, prange
from xtracer.log import Logger
import math

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


@jit(nopython=True, nogil=True, parallel=True)
def cal_bell_sas(xics):
    sas = np.zeros(len(xics))
    for i in prange(len(xics)):
        v = xics[i]
        norm_x = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2 +
                           v[4] ** 2 + v[5] ** 2 + v[6] ** 2) + 1e-6

        # y = np.array([0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044])
        norm_y = 0.531225
        s = v[0] * 0.0044 + v[1] * 0.054 + v[2] * 0.242 + v[3] * 0.399 + v[
            4] * 0.242 + v[5] * 0.054 + v[6] * 0.0044

        sa = s / (norm_x * norm_y)
        sa = min(sa, 1.)
        sa = 1 - 2 * math.acos(sa) / math.pi
        sas[i] = sa
    return sas


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


@jit(nopython=True, nogil=True, parallel=True)
def find_isotope_cluster(
        frame1_at, frame1_mz, frame1_height,
        idx_max_points, is_apex_v, xixs1,
        charge_min, charge_max, tol_iso_num,
        tol_at_area, tol_at_shift, tol_ppm, tol_pcc
):
    '''
    state_left_m是二维，因为只用看一个
    state_right_m是三维，因为可能需要同时考虑多个isotope
    state_gaussian_m：当左右都不存在，还要考察自身
    state is True when 1. 符合强度规律 2. 符合pcc阈值
    '''
    xix_gaussian = np.array(
        [0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044], dtype=np.float32
    )

    n_point = np.sum(is_apex_v)
    n_charge = charge_max - charge_min + 1
    apex_prefix = np.cumsum(is_apex_v) - 1

    state_left_m = np.zeros((n_point, n_charge), dtype=np.bool_)
    state_right_m = np.zeros((n_point, n_charge, tol_iso_num), dtype=np.bool_)
    state_lone_m = np.zeros((n_point, n_charge), dtype=np.bool_)

    for idx_i in prange(len(is_apex_v)): # 125
        if not is_apex_v[idx_i]:
            continue
        idx_apex = apex_prefix[idx_i]
        # if idx_apex == 15:
        #     a = 1

        i = idx_max_points[idx_i]
        i_at = frame1_at[i]
        i_mz = frame1_mz[i]
        i_xix = xixs1[idx_i]
        i_int = np.mean(i_xix[2:5])

        for charge in range(charge_min, charge_max + 1):
            # 先看M-1:
            target_mz = i_mz - C13_DELTA / charge
            limit_mz = target_mz * (1 - 50 * 1e-6)
            for idx_ii in range(idx_i - 1, -1, -1):
                ii = idx_max_points[idx_ii]
                ii_at = frame1_at[ii]
                ii_mz = frame1_mz[ii]
                ii_xix = xixs1[idx_ii]
                ii_int = np.mean(ii_xix[2:5])
                if ii_mz < limit_mz:
                    break
                if abs(ii_at - i_at) > tol_at_shift:
                    continue
                if ii_int < i_int * 1:
                    continue
                bias_ppm = 1e6 * abs(ii_mz - target_mz) / target_mz
                if bias_ppm > tol_ppm:
                    continue
                pcc = cal_pcc(i_xix, ii_xix)
                if pcc > tol_pcc:
                    state_left_m[idx_apex, charge-charge_min] = True
                    break

            # 如果有M-1, 不需要再看M+1
            if state_left_m[idx_apex, charge-charge_min]:
                continue

            # 再看M+1:
            found_right = False
            limit_mz = (i_mz + tol_iso_num * C13_DELTA / charge) * (1 + 50 * 1e-6)
            for idx_ii in range(idx_i + 1, len(idx_max_points)): # local maximum
                ii = idx_max_points[idx_ii]
                ii_at = frame1_at[ii]
                ii_mz = frame1_mz[ii]
                ii_xix = xixs1[idx_ii]
                ii_int = np.mean(ii_xix[2:5])
                if abs(ii_at - i_at) > tol_at_shift:
                    continue
                if ii_mz > limit_mz:
                    break
                if (ii_int > i_int) or (ii_int < 0.2 * i_int):
                    continue
                for n_neutron in range(1, tol_iso_num+1):
                    target_mz = i_mz + n_neutron * C13_DELTA / charge
                    bias_ppm = 1e6 * abs(ii_mz - target_mz) / target_mz
                    if bias_ppm > tol_ppm:
                        continue
                    pcc = cal_pcc(i_xix, ii_xix)
                    if pcc > tol_pcc:
                        state_right_m[idx_apex, charge-charge_min, n_neutron-1] = True
                        found_right = True
                        break

            # 无M-1, 无M+N
            if not found_right:
                pcc = cal_pcc(i_xix, xix_gaussian)
                if pcc > 0.75:
                    state_lone_m[idx_apex, charge-charge_min] = True

    return state_left_m, state_right_m, state_lone_m


def get_states(state_left_m, state_right_m, state_lone_m, allow_lone):
    '''
    单电荷规则: 左无右有，则该电荷有；其他则该电荷无
    跨电荷规则：左无右无，gaussian有，则全电荷有
    '''
    # Step 1: 单电荷规则
    final_m = (~state_left_m) & state_right_m

    # Step 2: 跨电荷规则
    # 选出那些单电荷规则后全 False 的行
    mask_all_false = ~final_m.any(axis=1)

    # 如果该行的 lone 全 True，则将整个行置 True
    mask_lone = mask_all_false & state_lone_m.all(axis=1)

    if allow_lone:
        final_m[mask_lone, 0] = True

    return final_m


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


from numba import jit
import numpy as np

@jit(nopython=True, nogil=True)
def dedup_local_max(
        mzs, ats, ints, cycles, tol_ppm, tol_at, tol_cycle
):

    n = len(mzs)
    keep = np.ones(n, dtype=np.uint8)
    right = 0

    for i in range(n):
        if keep[i] == 0:
            continue

        mz_i = mzs[i]
        at_i = ats[i]
        cycle_i = cycles[i]
        int_i = ints[i]

        mz_max = mz_i * (1 + tol_ppm * 1e-6)

        if right < i + 1:
            right = i + 1

        while right < n and mzs[right] <= mz_max:
            right += 1

        for j in range(i + 1, right):
            if keep[j] == 0:
                continue

            # 更快的 early reject
            if abs(cycles[j] - cycle_i) > tol_cycle:
                continue
            if abs(ats[j] - at_i) >= tol_at:
                continue

            int_j = ints[j]

            if int_j > int_i:
                keep[i] = 0
                break
            else:
                keep[j] = 0
    return keep


import numpy as np
from numba import njit


@jit(nogil=True, nopython=True, parallel=True)
def assign_mono_labels(
        mzs, ats, cycles, ints, tol_ppm, tol_im, tol_cycle
):
    n = mzs.shape[0]
    labels = np.zeros(n, dtype=np.int8)
    inv_mass = 1.00335

    for i in prange(n):
        # i = 10310
        mz_i = mzs[i]
        at_i = ats[i]
        cycle_i = cycles[i]
        int_i = ints[i]

        # -------------------------------
        # Step 2: 右峰 M+1 贪婪匹配
        # -------------------------------
        best_z = 0
        best_j_int = -1.0

        mz_max = mz_i + inv_mass * 1.1
        for j in range(i + 1, n):
            # j = 10736
            mz_j = mzs[j]
            int_j = ints[j]

            if mz_j > mz_max:
                break

            if abs(ats[j] - at_i) > tol_im or abs(cycles[j] - cycle_i) > tol_cycle:
                continue

            for z in range(1, 5):
                delta_mz = inv_mass / z
                bias_ppm = abs((mz_j - mz_i) - delta_mz) / mz_i * 1e6
                if bias_ppm <= tol_ppm:
                    best_z = z
                    break
                    # if int_j <= int_i:
                    #     if int_j > best_j_int:
                    #         best_j_int = int_j
                    #         best_z = z
        labels[i] = best_z

    return labels