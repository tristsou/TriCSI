TriCSI CSI CSV parser patch v3 (2026-10-30)

Why: original headered data can store 128 IQ values per packet while 'len' records the full raw 384-value Wi-Fi CSI vector. Previous parser mistakenly required saved IQ column count to equal full 'len'. This can discard valid rows.

Contents:
  TriCSI_One_Day_Code/common.py                  -- updated parser
  TriCSI_One_Day_Code/00_discover_and_audit.py   -- audited version
  TriCSI_One_Day_Code/05_check_csv_layout.py     -- new read-only layout diagnostic

Installation:
  cd C:\ESP32-CSI-Tool\tricsi_localization
  Expand-Archive '.\TriCSI_Parser_Patch_v3.zip' -DestinationPath . -Force
  $K = '.\TriCSI_One_Day_Code'
  $Py = '.\.venv\Scripts\python.exe'

First:
  & $Py "$K\05_check_csv_layout.py" --root '.\data_3tx\raw' --sessions 3TX_S1 3TX_S2 3TX_S3

Then:
  & $Py "$K\00_discover_and_audit.py" --root '.\data_3tx\raw' --sessions 3TX_S1 3TX_S2 3TX_S3 --output "$K\three_link_qc_v3.csv"

Please compare total_valid_packets against historical S1=40,644 and S2=40,649. Disagreements require investigation before training. Do not overwrite historical raw data or saved models.

This is a CSV parser fix only. The 'original 52 subcarrier indices' and 'seven packet windows per trial' used in the original analysis remain UNVERIFIED in the reconstructed feature extraction. Do not train new models until that parity is resolved.
