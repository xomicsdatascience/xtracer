#!/usr/bin/env python3

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sdk import mbisdk
from pathlib import Path
from sklearn.linear_model import LinearRegression, RANSACRegressor
from matplotlib.colors import LinearSegmentedColormap

C13_DELTA = 1.0033548378
import mmap
import re

from pyteomics import mgf

class SimpleMGF:
    def __init__(self, path):
        self.reader = mgf.IndexedMGF(str(path))

    def get_rt(self, title):
        """返回保留时间（秒），若不存在返回 None"""
        try:
            spec = self.reader[title]
        except KeyError:
            return None
        params = spec.get('params', {})
        rt = params.get('rtinseconds') or params.get('RTINSECONDS')
        return float(rt) if rt is not None else None

    def get_peaks(self, title):
        """返回 (mz, intensity) 数组，形状 (n,2)"""
        try:
            spec = self.reader[title]
        except KeyError:
            return np.empty((0, 2))
        mz = spec.get('m/z array', [])
        intensity = spec.get('intensity array', [])
        return mz, intensity

    def construct_title_at_dict(self):
        title_at_d = {}
        for title in self.reader.index:          # 遍历所有标题
            try:
                spec = self.reader[title]
            except KeyError:
                continue
            at_value = spec.get('params', {}).get('at')
            title_at_d[title] = float(at_value)
        return title_at_d

    def close(self):
        self.reader.close()

def extract_at_from_mgf(path):
    # 匹配规则：TITLE=xxx\n ... AT=xxx\n （非贪婪跨行）
    # 实际测试中，此正则在标准 MGF 上稳定且极快
    pattern = re.compile(
        rb'TITLE=(?P<title>[^\n]+)\n(?:.*\n)*?AT=(?P<at>[^\n]+)', re.MULTILINE)
    spectra = {}

    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for match in pattern.finditer(mm):
                # mmap 返回 bytes，需解码并转 float
                spectra[match.group('title').decode()] = float(
                    match.group('at'))
    return spectra


@st.cache_data
def load_and_merge(fmgf, fsage, fmatch):
    mymgf = SimpleMGF(fmgf)
    # scannr_at_d = mymgf.construct_title_at_dict()
    scannr_at_d = extract_at_from_mgf(fmgf)

    df_sage = pd.read_csv(fsage, sep='\t')
    df_sage = df_sage[
        (df_sage['label'] == 1) &
        (df_sage['peptide_q'] < 0.01) &
        (df_sage['filename'] == fmgf.name)
    ]
    df_sage = df_sage.reset_index(drop=True)

    df_sage['pr_id'] = df_sage['peptide'] + df_sage['charge'].astype(str)
    df_sage = df_sage.sort_values('peptide_q', ascending=True)
    df_sage = df_sage.drop_duplicates('pr_id', keep='first')
    df_sage = df_sage.reset_index(drop=True)

    PROTON = 1.007276466812
    df_sage['pr_mz'] = (df_sage['calcmass'] + df_sage['charge'] * PROTON) / df_sage['charge']
    df_sage['pr_rt'] = df_sage['rt'] * 60.

    # 从mgf找淌度
    df_sage['scannr'] = df_sage['scannr'].astype(str)
    assert df_sage['scannr'].isin(scannr_at_d.keys()).all()
    df_sage['pr_at'] = df_sage['scannr'].map(scannr_at_d)

    # 从matched找fg_mz, fg_anno
    df_fg = pd.read_csv(fmatch, sep='\t')
    df_fg = df_fg[df_fg['psm_id'].isin(df_sage['psm_id'])].reset_index(drop=True)
    df_fg = df_fg.sort_values(['psm_id', 'fragment_intensity'], ascending=[True, False])
    df_fg['fragment_mz_calculated'] = df_fg['fragment_mz_calculated'].astype(str)
    df_fg['fragment_intensity'] = df_fg['fragment_intensity'].astype(str)
    df_fg['fragment_ordinals'] = df_fg['fragment_ordinals'].astype(str)
    df_fg['fragment_charge'] = df_fg['fragment_charge'].astype(str)
    df_fg['fg_anno'] = df_fg['fragment_type'] + df_fg['fragment_ordinals'] + '_' + df_fg['fragment_charge']
    df_fg = df_fg.groupby('psm_id', sort=False).agg(
        fg_mz=('fragment_mz_calculated', ';'.join),
        fg_anno=('fg_anno', ';'.join),
        fg_int=('fragment_intensity', ';'.join)
    ).reset_index()
    df_fg = df_fg[['psm_id', 'fg_mz', 'fg_anno', 'fg_int']]

    df = pd.merge(df_sage, df_fg, on="psm_id")

    cols = ['psm_id', 'scannr', 'pr_id', 'pr_mz', 'pr_rt', 'pr_at', 'fg_mz', 'fg_anno', 'fg_int']
    df = df[cols]
    return df, mymgf


# 缓存 MBIFile 初始化，避免重复 Init 调用
@st.cache_resource
def load_mbi_file(mbi_path):
    mbi = mbisdk.MBIFile()
    mbi.SetFilename(str(mbi_path))
    mbi.Init()
    return mbi


def init_gui():
    st.set_page_config(page_title="xTracer Viewer", layout="wide")
    st.markdown("""
            <style>
                .block-container {
                    padding-top: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)


def get_frame_data(mbi, frame_i):
    frame = mbi.GetFrame(int(frame_i))
    frame_at, frame_mz, frame_height = [], [], []
    for i_scan in frame.GetNonZeroScanIndices():
        scan_at = frame.GetArrivalBinTimeOffset(i_scan)
        scan = frame.GetMassSpectrum(i_scan)
        scan_mz = scan.mz
        scan_height = scan.intensities
        frame_at.extend([scan_at] * len(scan_mz))
        frame_mz.extend(scan_mz)
        frame_height.extend(scan_height)
    frame_at = np.array(frame_at).astype(np.float32)
    frame_mz = np.array(frame_mz).astype(np.float32)
    frame_height = np.array(frame_height)

    # 排序
    idx = np.argsort(frame_mz)
    frame_at = frame_at[idx]
    frame_mz = frame_mz[idx]
    frame_height = frame_height[idx]
    return frame_at, frame_mz, frame_height


def get_area_raw_data(frame_at, frame_mz, frame_y, at, mz, tol_at, tol_ppm):
    mz_min = mz * (1 - tol_ppm * 1e-6)
    mz_max = mz * (1 + tol_ppm * 1e-6)
    at_min, at_max = at - tol_at, at + tol_at
    idx = ((frame_at < at_max) &
           (frame_at > at_min) &
           (frame_mz < mz_max) &
           (frame_mz > mz_min))
    ats, mzs, ys = frame_at[idx], frame_mz[idx], frame_y[idx]
    ppms = 1e6 * (mzs - mz) / mz
    return ats, mzs, ys, ppms


def get_big_area_raw_data(frame_at, frame_mz, frame_y, at_min, at_max, mz_min, mz_max):
    idx = ((frame_at < at_max) &
           (frame_at > at_min) &
           (frame_mz < mz_max) &
           (frame_mz > mz_min))
    ats, mzs, ys = frame_at[idx], frame_mz[idx], frame_y[idx]
    return ats, mzs, ys


def get_area_merge_data(frame_at, frame_mz, frame_y, at, mzs, tol_at, tol_ppm):
    # 将一个area的信号merge，形成一个时刻强度，为XIC作准备
    y_v = []
    at_min, at_max = at - tol_at, at + tol_at
    for mz in mzs:
        mz_min = mz * (1 - tol_ppm * 1e-6)
        mz_max = mz * (1 + tol_ppm * 1e-6)
        idx = ((frame_at < at_max) &
               (frame_at > at_min) &
               (frame_mz < mz_max) &
               (frame_mz > mz_min))
        y_v.append(frame_y[idx].sum())
    return np.array(y_v)


def smooth_vec_np(vec):
    if vec.ndim == 1:
        vec_smooth = np.empty_like(vec)
        vec_smooth[0] = 0.667 * vec[0] + 0.333 * vec[1]
        vec_smooth[-1] = 0.667 * vec[-1] + 0.333 * vec[-2]
        vec_smooth[1:-1] = 0.25 * vec[:-2] + 0.5 * vec[1:-1] + 0.25 * vec[2:]
        return vec_smooth
    elif vec.ndim == 2:
        vec_smooth = np.empty_like(vec)
        vec_smooth[:, 0] = 0.667 * vec[:, 0] + 0.333 * vec[:, 1]
        vec_smooth[:, -1] = 0.667 * vec[:, -1] + 0.333 * vec[:, -2]
        vec_smooth[:, 1:-1] = 0.25 * vec[:, :-2] + 0.5 * vec[:, 1:-1] + 0.25 * \
                              vec[:, 2:]
        return vec_smooth


def get_xics(mbi, frame_cidx, at, mzs, tol_at, tol_ppm, total_cycle):
    half_cycle_num = int((total_cycle - 1) / 2)
    frame_i_left = frame_cidx - half_cycle_num*2
    frame_i_right = frame_cidx + half_cycle_num*2
    rts, xics = [], []
    for frame_i in range(frame_i_left, frame_i_right+1, 2):
        rts.append(mbi.GetRetentionTimes()[frame_i])
        frame_at, frame_mz, frame_y = get_frame_data(mbi, frame_i)
        y_v = get_area_merge_data(
            frame_at, frame_mz, frame_y, at, mzs, tol_at, tol_ppm
        )
        xics.append(y_v)
    xics = np.array(xics).T
    xics = smooth_vec_np(xics)
    return np.array(rts), xics


def get_xims(mbi, frame_cidx, at_target, mzs_target,
             tol_at, tol_ppm,
             merge_num):
    n_bins = 23
    half_cycle_num = int((merge_num - 1) / 2)
    frame_i_left = frame_cidx - half_cycle_num*2
    frame_i_right = frame_cidx + half_cycle_num*2
    bin_width = 2 * tol_at / n_bins
    im_start = at_target - tol_at
    result_ats = np.linspace(at_target - tol_at + bin_width / 2,
                             at_target + tol_at - bin_width / 2,
                             n_bins)
    result_xims = np.zeros((len(mzs_target), n_bins))

    for frame_i in range(frame_i_left, frame_i_right+1, 2):
        frame_at, frame_mz, frame_y = get_frame_data(mbi, frame_i)
        for i_mz, mz in enumerate(mzs_target):
            area_ats, area_mzs, area_ys, _ = get_area_raw_data(
                frame_at, frame_mz, frame_y, at_target, mz, tol_at, tol_ppm
            )
            for i in range(len(area_ats)):
                point_at, point_y = area_ats[i], area_ys[i]
                bin_idx = int((point_at - im_start) / bin_width)
                if 0 <= bin_idx < n_bins:
                    result_xims[i_mz, bin_idx] += point_y
    result_xims = smooth_vec_np(result_xims)
    return result_ats, result_xims


def get_apex_frame_idx(mbi, row):
    frames_rt = np.array(mbi.GetRetentionTimes())
    frames_level = np.array(mbi.GetFrameMSLevels())
    rt = row['pr_rt']
    frame_i = (abs((frames_rt - rt) + (frames_level != 2) * 1e9)).argmin()
    return frame_i


def plot_xics(mbi, row, tol_at, tol_ppm, total_cycle, rt_shift):
    # 确定pr的rt最接近的frame，画ms1和ms2在该frame上的原始信号
    frames_rt = np.array(mbi.GetRetentionTimes())
    frames_level = np.array(mbi.GetFrameMSLevels())
    pr_id, pr_mz = row['pr_id'], row['pr_mz']
    pr_charge = int(pr_id[-1])
    rt, at_measure = row['pr_rt'], row['pr_at']
    rt += rt_shift
    fg_mzs = np.fromstring(row['fg_mz'], sep=';')
    fg_annos = row['fg_anno'].split(';')

    frame_cidx = (abs((frames_rt - rt) + (frames_level != 2) * 1e9)).argmin()

    fig, ax = plt.subplots(2, 1, sharex=True)
    # tol_at, tol_ppm = 1.5, 25 # 2.5, 30

    # M, M+1, M+2
    labels = ['M', 'M+1', 'M+2']
    mzs = []
    for neutron_i in range(3):
        mz = pr_mz + neutron_i * C13_DELTA / pr_charge
        mzs.append(mz)
    N = total_cycle
    rts, xics = get_xics(mbi, frame_cidx, at_measure, mzs, tol_at, tol_ppm, N)
    for i in range(xics.shape[0]):
        ax[0].plot(rts, xics[i], label=labels[i])
    ax[0].legend(loc='best')
    ax[0].ticklabel_format(style='plain', axis='y')

    # unfrag
    # mzs = np.concatenate([[pr_mz], fg_mzs])
    # rts, xics = get_xics(mbi, frame_cidx+1, at_measure, mzs, tol_at, tol_ppm, N)
    # ax[1].plot(rts, xics[0], label='Unfrag')
    # ax[1].legend(loc='best')

    # fg
    rts, xics = get_xics(mbi, frame_cidx+1, at_measure, fg_mzs, tol_at, tol_ppm, N)
    for i in range(xics.shape[0]):
        ax[1].plot(rts, xics[i], label=f'{fg_annos[i]}')
    ax[1].legend(loc='best')
    fig.supxlabel("RT (s)")
    plt.tight_layout()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
        # st.pyplot(fig, width='content')


def plot_xims(mbi, row, tol_at, tol_ppm, merge_num, rt_shift):
    # 确定pr的rt最接近的frame，画ms1和ms2在该frame上的原始信号
    frames_rt = np.array(mbi.GetRetentionTimes())
    frames_level = np.array(mbi.GetFrameMSLevels())
    pr_id, pr_mz = row['pr_id'], row['pr_mz']
    pr_charge = int(pr_id[-1])
    rt, at_measure = row['pr_rt'], row['pr_at']
    rt += rt_shift
    fg_mzs = np.fromstring(row['fg_mz'], sep=';')
    fg_annos = row['fg_anno'].split(';')

    frame_cidx = (abs((frames_rt - rt) + (frames_level != 2) * 1e9)).argmin()

    fig, ax = plt.subplots(2, 1, sharex=True)
    # tol_at, tol_ppm = 1.5, 25 # 2.5, 30

    # M, M+1, M+2
    labels = ['M', 'M+1', 'M+2']
    mzs = []
    for neutron_i in range(3):
        mz = pr_mz + neutron_i * C13_DELTA / pr_charge
        mzs.append(mz)
    N = merge_num
    ats, xims = get_xims(mbi, frame_cidx, at_measure, mzs, tol_at, tol_ppm, N)
    for i in range(xims.shape[0]):
        ax[0].plot(ats, xims[i], label=labels[i])
    ax[0].legend(loc='best')
    ax[0].ticklabel_format(style='plain', axis='y')

    # unfrag
    # mzs = np.concatenate([[pr_mz], fg_mzs])
    # ats, xims = get_xims(mbi, frame_cidx+1, at_measure, mzs, tol_at, tol_ppm, N)
    # ax[1].plot(ats, xims[0], label='Unfrag')
    # ax[1].legend(loc='best')

    # fg
    ats, xims = get_xims(mbi, frame_cidx+1, at_measure, fg_mzs, tol_at, tol_ppm, N)
    for i in range(xims.shape[0]):
        ax[1].plot(ats, xims[i], label=fg_annos[i])
    ax[1].legend(loc='best')
    fig.supxlabel("AT (ms)")
    plt.tight_layout()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
        # st.pyplot(fig, width='content')


def plot_heatmap(mbi, row, tol_at, tol_da_left, tol_da_right, rt_shift):
    frames_rt = np.array(mbi.GetRetentionTimes())
    frames_level = np.array(mbi.GetFrameMSLevels())

    # 确定pr的rt最接近的frame，画ms1和ms2在该frame上的原始信号
    pr_id, pr_mz = row["pr_id"], row['pr_mz']
    pr_charge = int(pr_id[-1])
    rt, at = row['pr_rt'], row['pr_at']
    rt += rt_shift
    fg_mzs = np.fromstring(row['fg_mz'], sep=';')
    fg_annos = row['fg_anno'].split(';')

    frame_cidx = (abs((frames_rt - rt) + (frames_level != 2) * 1e9)).argmin()

    frame1_at, frame1_mz, frame1_y = get_frame_data(mbi, frame_cidx)
    frame2_at, frame2_mz, frame2_y = get_frame_data(mbi, frame_cidx + 1)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6, 3))

    # M-1, M, M+1, M+2
    mz_min, mz_max = pr_mz - tol_da_left, pr_mz + tol_da_right
    at_min, at_max = at - tol_at, at + tol_at
    ats, mzs, ints = get_big_area_raw_data(frame1_at, frame1_mz, frame1_y, at_min, at_max, mz_min, mz_max)
    assert len(ats) > 0
    colors = [
        'black', '#0000AA', '#0088FF', '#00FF88',
        '#88FF00', '#FFFF00', '#FF8800', '#FF0000'
    ]
    cmap = LinearSegmentedColormap.from_list('ms_custom', colors, N=256)
    nx, ny = int((mz_max - mz_min) / 0.01), int((at_max - at_min) / 0.1)
    H, xedges, yedges = np.histogram2d(
        mzs, ats, weights=ints, bins=(nx, ny),
        range=[[mz_min, mz_max], [at_min, at_max]]
    )
    vmax = np.percentile(H[H > 0], 98) if np.any(H > 0) else H.max()
    vmax = max(vmax, 1)  # 确保vmax至少为1
    ax.imshow(H.T, origin='lower', aspect='auto',
               extent=[mz_min, mz_max, at_min, at_max],
               cmap=cmap, vmin=0, vmax=vmax)
    ax.set_ylabel('AT (ms)')
    ax.set_xlabel('m/z')
    ax.tick_params(axis='both', direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
        # st.pyplot(fig, width='content')


def plot_psm(mymgf, row):
    # 确定pr的rt最接近的frame，画ms1和ms2在该frame上的原始信号
    scannr = row['scannr']

    # 从mgf中读取peaks
    peaks_mz, peaks_int = mymgf.get_peaks(scannr)
    color_dict = {'b': 'blue', 'y': 'red', 'by': 'purple', 'none': 'black'}

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    labels = ['none'] * len(peaks_mz)
    for i in range(len(peaks_mz)):
        ax.vlines(peaks_mz[i], 0, peaks_int[i], color=color_dict[labels[i]], linewidth=1.5)

    # 从matched中读取peaks
    fg_mzs = np.fromstring(row['fg_mz'], sep=';')
    fg_annos = row['fg_anno'].split(';')
    fg_ints = np.fromstring(row['fg_int'], sep=';')
    labels = [fg_anno[0] for fg_anno in fg_annos]
    for i in range(len(fg_mzs)):
        ax.vlines(fg_mzs[i], 0, fg_ints[i], color=color_dict[labels[i]], linewidth=1.5)
        if labels[i] in ['b', 'by']:
            ax.text(fg_mzs[i], fg_ints[i] + 20, fg_annos[i], color='blue',
                     ha='center', va='bottom', rotation=90)
        if labels[i] in ['y', 'by']:
            ax.text(fg_mzs[i], fg_ints[i] + 20, fg_annos[i], color='red',
                    ha='center', va='bottom', rotation=90)
    ax.set_ylim(0, max(peaks_int) * 1.2)
    ax.set_ylabel('Intensity')
    ax.set_xlabel('m/z')
    plt.tight_layout()

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)
        # st.pyplot(fig, width='content')


def increment():
    st.session_state.frame_idx += 2

def decrement():
    st.session_state.frame_idx -= 2


def main(fmbi, fmgf, fsage, fmatch):
    init_gui()

    # 各种路径
    st.sidebar.write(".MBI path:")
    st.sidebar.write(fmgf.name)
    st.sidebar.write(".MGF path:")
    st.sidebar.write(fmgf.name)
    # st.sidebar.write("Sage path:", fsage.name)
    # st.sidebar.write("Sage matched path:", fmatch.name)

    # 初始化并缓存 .mbi 文件对象
    mbi = load_mbi_file(fmbi.resolve())

    # 下拉列表切换绘图种类
    st.sidebar.title("Select Plot Type")
    plot_options = ['XIC', 'XIM', 'Heatmap', 'MS/MS']
    st.session_state.selected_plot = 'XIC'
    st.session_state.selected_plot = st.sidebar.selectbox(
        "Plot Type", plot_options, index=plot_options.index(st.session_state.selected_plot)
    )

    # 加载mgf/sage/matched.tsv，并把离子m/z和at拼接上去
    df_merged, mymgf = load_and_merge(fmgf, fsage, fmatch)
    # st.subheader(f'Sage Report: {len(df_merged)} prs')
    st.markdown(
        f'<div style="margin: 0.8rem 0 0.5rem 0;">'
        f'<span style="font-size: 1.25rem; font-weight: 600; '
        f'line-height: 1.8; padding: 0.4rem 0; display: inline-block;">'
        f'Sage Report: {len(df_merged)} prs</span></div>',
        unsafe_allow_html=True
    )
    st.dataframe(df_merged, height=200, width='stretch')

    # 选择并显示选中行详情
    if "selected_pr" not in st.session_state:
        st.session_state.selected_pr = df_merged['pr_id'].iloc[0]
    selected_pr = st.selectbox(
        "Selected pr", df_merged['pr_id'].unique().tolist(), key="selected_pr"
    )
    selected_row = df_merged[df_merged['pr_id'] == selected_pr]
    st.dataframe(selected_row, width='stretch')
    selected_row = selected_row.iloc[0]
    # st.markdown("---")  # 分割线

    # Initialize frame_idx when selected_pr changes
    if 'last_selected_pr' not in st.session_state or st.session_state.last_selected_pr != st.session_state.selected_pr:
        st.session_state.frame_idx = get_apex_frame_idx(mbi, selected_row)
        # Reset number_input defaults on peptide change
        st.session_state['top_k'] = 100
        st.session_state['xic_tol_ppm'] = 30
        st.session_state['xic_tol_at'] = 2.
        st.session_state['xic_total_cycle'] = 13
        st.session_state['xic_rt_shift'] = 0
        st.session_state['xim_tol_ppm'] = 30
        st.session_state['xim_tol_at'] = 2.
        st.session_state['merge_num'] = 1
        st.session_state['heat_tol_da_left'] = 1.5
        st.session_state['heat_tol_da_right'] = 3.
        st.session_state['heat_tol_at'] = 2.
        st.session_state.last_selected_pr = st.session_state.selected_pr
    
    # 绘图区
    if st.session_state.selected_plot == "XIC":
        tol_ppm = st.sidebar.number_input(
            "tol_ppm (ppm)", min_value=5, max_value=50,
            step=2, value=30,
            key='xic_tol_ppm'
        )
        tol_at = st.sidebar.number_input(
            "tol_at (ms)", min_value=0.0, max_value=10.0,
            step=0.5, value=2.,
            key='xic_tol_at'
        )
        total_cycle = st.sidebar.number_input(
            "total cycles", min_value=7, max_value=33,
            step=2, value=13,
            key='xic_total_cycle'
        )
        st.session_state.rt_shift = st.sidebar.number_input(
            "rt shift", min_value=-20, max_value=20,
            step=2, value=0,
            key='xic_rt_shift'
        )
        plot_xics(mbi, selected_row, tol_at, tol_ppm, total_cycle, st.session_state.rt_shift)

    if st.session_state.selected_plot == "XIM":
        tol_ppm = st.sidebar.number_input(
            "tol_ppm (ppm)", min_value=0, max_value=100,
            step=5, value=30,
            key='xim_tol_ppm'
        )
        tol_at = st.sidebar.number_input(
            "tol_at (ms)", min_value=0.0, max_value=10.0,
            step=0.5, value=2.,
            key='xim_tol_at'
        )
        merge_num = st.sidebar.number_input(
            "merged nums", min_value=1, max_value=7,
            step=2, value=1,
            key='merge_num'
        )
        plot_xims(mbi, selected_row, tol_at, tol_ppm, merge_num, st.session_state.rt_shift)

    if st.session_state.selected_plot == "Heatmap":
        tol_da_left = st.sidebar.number_input(
            "tol_da_left", min_value=0., max_value=5.,
            step=0.5, value=1.5,
            key='heat_tol_da_left'
        )
        tol_da_right = st.sidebar.number_input(
            "tol_da_right", min_value=0., max_value=5.,
            step=0.5, value=3.,
            key='heat_tol_da_right'
        )
        tol_at = st.sidebar.number_input(
            "tol_at (ms)", min_value=0.0, max_value=5.0,
            step=0.5, value=2.,
            key='heat_tol_at'
        )
        plot_heatmap(mbi, selected_row, tol_at, tol_da_left, tol_da_right, st.session_state.rt_shift)

    if st.session_state.selected_plot == 'MS/MS':
        plot_psm(mymgf, selected_row)


import sys
if __name__ == "__main__":
    # fmbi = Path(sys.argv[1])
    # fmgf = Path(sys.argv[2])
    # fsage = Path(sys.argv[3])

    fmbi = Path(r"D:\Jesse\xtracer\data_mbi3\2024-12-14 06.20.30-SSL_5_8-updated_KO.mbi")
    fmgf = Path(r"D:\Jesse\xtracer\data_mbi3\xix\2024-12-14 06.20.30-SSL_5_8-updated_KO.mgf")
    fsage = Path(r"D:\Jesse\xtracer\data_mbi3\sage_results\results.sage.tsv")

    fmatch = Path(fsage).parent/'matched_fragments.sage.tsv'
    main(fmbi, fmgf, fsage, fmatch)