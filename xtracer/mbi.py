import numpy as np

from collections import deque
from xtracer.log import Logger

try:
    from xtracer.sdk import mbisdk
except ImportError:
    raise RuntimeError(
        "SDK to read .mbi files is required. "
        "Please email support@mobilionsystems.com to request SDK. "
        "Then, copy the _mbisdk.pyd, MBI_SDK.dll and mbisdk.py to the sdk folder"
    )

try:
    profile
except:
    profile = lambda x: x

logger = Logger.get_logger()

class MBIReader:
    def __init__(self, fdir, across_cycle_num):
        self.mbi = mbisdk.MBIFile(str(fdir))
        if self.mbi.Init():
            logger.info(f'{str(fdir.name)} load successfully.')
        else:
            logger.info('.mbi load failed.')
        self.cal = self.mbi.GetCalibration()
        self.mz_lookup = self.build_mz_lookup()

        self.deque_frame1 = deque(maxlen=across_cycle_num)
        self.deque_frame2 = deque(maxlen=across_cycle_num)

        self.cycle_frame_num = 2
        self.frame_expand_num = int(across_cycle_num / 2)


    def __getattr__(self, name):
        # can use raw function without wrapper
        return getattr(self.mbi, name)


    @profile
    def get_frame_data(self, frame_idx):
        frame = self.mbi.GetFrame(frame_idx)
        csr_tuple = frame.GetFrameDataAsCSRComponents()
        _, frame_height, indices_vec, indptr_vec = csr_tuple
        frame_height = np.array(frame_height, dtype=np.float32)
        indices = np.array(indices_vec, dtype=np.int64)
        indptr = np.array(indptr_vec, dtype=np.int64)
        non_zero_scans = np.array(frame.GetNonZeroScanIndices(), dtype=np.int64)

        row_count = indptr.size - 1
        total_scans = frame.GetNumScans()
        if row_count == total_scans:
            scan_indices = np.arange(total_scans, dtype=np.int64)
        elif row_count == non_zero_scans.size:
            scan_indices = non_zero_scans
        else:
            frame.Unload()
            raise ValueError(f'CSR error: rows={row_count}, '
                             f'non_zero={non_zero_scans.size}')

        # arrival times
        counts = np.diff(indptr)
        arrival_offsets = np.array(
            [frame.GetArrivalBinTimeOffset(int(scan_idx)) for scan_idx in
             scan_indices], dtype=np.float32
        )
        frame_at = np.repeat(arrival_offsets, counts)

        # idx --> m/z
        frame_mz = self.mz_lookup[indices]

        # ascending sort by m/z
        idx = np.argsort(frame_mz)
        frame_at = np.ascontiguousarray(frame_at[idx])
        frame_mz = np.ascontiguousarray(frame_mz[idx])
        frame_height = np.ascontiguousarray(frame_height[idx])

        # unload
        frame.Unload()

        return frame_at, frame_mz, frame_height


    def build_mz_lookup(self, max_idx=500000):
        # idx --> m/z
        indices = np.arange(max_idx, dtype=np.uint64)
        return np.array(
            [self.cal.IndexToMz(int(i)) for i in indices],
            dtype=np.float32
        )


    @profile
    def load_frames_to_deque(self, idx):
        if len(self.deque_frame1) == 0:  # loop start
            for i in range(-self.frame_expand_num,
                           self.frame_expand_num + 1):
                frame_idx = idx + i * self.cycle_frame_num
                self.deque_frame1.append(self.get_frame_data(frame_idx))
                self.deque_frame2.append(self.get_frame_data(frame_idx+1))
        else:
            frame_idx = idx + self.frame_expand_num * self.cycle_frame_num
            self.deque_frame1.append(self.get_frame_data(frame_idx))
            self.deque_frame2.append(self.get_frame_data(frame_idx + 1))
