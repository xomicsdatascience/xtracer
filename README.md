# xTracer

Parallel Accumulation with Mobility Aligned Fragmentation ([PAMAF](https://www.biorxiv.org/content/biorxiv/early/2024/10/22/2024.10.18.619158.full.pdf)) achieves near-complete ion utilization and high spectral specificity by fragmenting all mobility-separated precursors without quadrupole isolation. Leveraging the ultrahigh mobility resolution of SLIM, this quadrupole-free strategy maximizes ion sampling efficiency and offers a promising approach in mass spectrometry–based proteomics, particularly for low-abundance peptides or low-input samples. However, the unique data structure of PAMAF—where precursor–fragment relationships are encoded along the mobility dimension—renders it incompatible with existing peptide identification tools. Here, we present xTracer, the first untargeted peptide identification algorithm developed specifically for PAMAF data. xTracer integrates correlations across both chromatographic and mobility dimensions to associate precursor and fragment ions, reconstruct pseudo-spectra, and enable database searching using well-established DDA search engines. Applied to datasets with varying sample loads and acquisition throughputs, xTracer consistently achieved robust and reproducible peptide identifications, outperforming single-domain correlation strategies. Overall, xTracer provides a versatile and high-efficiency computational framework for reconstructing pseudo-spectra from quadrupole-free, mobility-aligned fragmentation data, enhancing the analytical power of high-resolution ion mobility–based proteomics.

---
### Contents
**[Datasets](#dataset)**<br>
**[Installation](#installation)**<br>
**[Usage](#usage)**<br>
**[Output](#output)**<br>

---
### Datasets
Varying sample load dataset and varying throughput dataset by PAMAF acquisition can be downloaded from [MSV000099577](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000099577
)

### Installation
We recommend using [Conda](https://www.anaconda.com/) to create a Python environment for using xTracer on Windows.

1. Create a Python environment with version 3.12.11 to consistent with the SDK environment.
    ```bash
    conda create -n xtracer_env python=3.12.11
    conda activate xtracer_env
    ```

2. Install xTracer
    ```bash
    pip install xtracer
    ```

3. SDK access
>   Please send an SDK request email to [Mobilion Inc.](mailto:daniel.debord@mobilionsystems.com), 
>   and then copy the file *_mbisdk.pyd, MBI_SDK.dll and mbisdk.py* into the sdk folder under the xTracer installation directory.

---
### Usage
```bash
xtracer -ws_in "the folder that contains .mbi files" -xic -xim
```
All params are list below by entering `xtracer -h`:
```
optional arguments for users:
  -h, --help                     Show this help message and exit.
  -ws_in WS_IN                   Specify the folder that contains .mbi files.
  -out_name OUT_NAME             Specify the folder name that contains .mgf files. Default: mgf_xtracer
  -xic                           Using XIC-based method to calculate PCC
  -xim                           Using XIM-based method to calculate PCC
  -pr_mz_min PR_MZ_MIN           Specify the minimum m/z value of precursors. Default: 200
  -charge_min CHARGE_MIN         Specify the minimum charge of precursors. Default: 1
  -charge_max CHARGE_MAX         Specify the maximum charge of precursors. Default: 4
  -at_min AT_MIN                 Specify the minimum arrival time (at) value of signals. Default: 90 ms
  -tol_at_area TOL_AT_AREA       Specify the millisecond tolerance of signal in at dimension. Default: 2.5
  -tol_at_shift TOL_AT_SHIFT     Specify the millisecond tolerance when considering signal related. Default: 1
  -tol_ppm TOL_PPM               Specify the ppm tolerance of signal in m/z dimension. Default: 30
  -tol_iso_num TOL_ISO_NUM       Specify how many isotopes should have to be a precursor. Default: 2, i.e. M, M+1H, M+2H
  -tol_pcc TOL_PCC               Specify the PCC tolerance when two signal are related. Default: 0.3
  -tol_point_num TOL_POINT_NUM   Specify the point num tolerance that a signal should have. Default: 9
  -tol_fg_num TOL_FG_NUM         Specify the fragment ions num tolerance that a spectrum should have. Default: 8
  -xim_across_cycle_num          Specify the odd XIM cycle span when summing frames. Default: 3
  -xic_across_cycle_num          Specify the odd XIC cycle span when extracting XIC. Default: 7
```

### Output
For each .mbi file, xTracer produces a corresponding .mgf DDA-like file that can be analyzed by DDA engines for identification.

---
## Troubleshooting
- Please create a GitHub issue and we will respond as soon as possible.
- Email: jian.song.2025@outlook.com

---