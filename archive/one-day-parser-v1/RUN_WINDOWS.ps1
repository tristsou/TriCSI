# Execute step-by-step, not all at once. Copy this file and adjust $Root and $DataRoot first.
$Root = 'C:\ESP32-CSI-Tool\tricsi_localization'
$Kit = Join-Path $Root 'TriCSI_One_Day_Code'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
Set-Location $Root

# A. READ-ONLY local code & model inventory.
& $Python "$Kit\00_project_inventory.py" --root $Root --output "$Kit\project_inventory.txt"

# B. Discover transmitter MACs and inspect trial quality on development data.
# The uploaded project stores the session directories in data_3tx\raw.
& $Python "$Kit\00_discover_and_audit.py" --root "$Root\data_3tx\raw" --sessions 3TX_S1 3TX_S2 3TX_S3 --output "$Kit\three_link_qc.csv"

# C. EDIT "$Kit\config.json" with real MACs and verify 52-index mapping vs your old extractor.
# Then uncomment in the order shown. Do not fit anything on S3/S4/S5.

# & $Python "$Kit\01_build_features.py" --config "$Kit\config.json" --sessions 3TX_S1 3TX_S2 --output "$Kit\dev_features.npz"
# & $Python "$Kit\02_train_freeze.py" --features "$Kit\dev_features.npz" --out "$Kit\frozen_models"
# & $Python "$Kit\01_build_features.py" --config "$Kit\config.json" --sessions 3TX_S3 --output "$Kit\S3_features.npz"
# & $Python "$Kit\03_evaluate_frozen.py" --models "$Kit\frozen_models" --features "$Kit\S3_features.npz" --sessions 3TX_S3 --out "$Kit\S3_reconstructed_predictions.csv"
# & $Python "$Kit\04_analyze_and_plot.py" --predictions "$Kit\S3_reconstructed_predictions.csv" --out "$Kit\S3_parity_check"

# AFTER approvals and AFTER model freezing, collect new sessions with YOUR ORIGINAL COLLECTOR.
# AFTER raw session CSVs exist, uncomment:
# & $Python "$Kit\01_build_features.py" --config "$Kit\config.json" --sessions 3TX_S4 3TX_S5 --output "$Kit\new_features.npz"
# & $Python "$Kit\03_evaluate_frozen.py" --models "$Kit\frozen_models" --features "$Kit\new_features.npz" --sessions 3TX_S4 3TX_S5 --out "$Kit\S4_S5_predictions.csv"
# & $Python "$Kit\04_analyze_and_plot.py" --predictions "$Kit\S4_S5_predictions.csv" --out "$Kit\new_results"

# For OLD TWO-LINK sessions, locate data directory and run separately:
# & $Python "$Kit\00_discover_and_audit.py" --root "C:\PATH\TO\TWO_LINK_SESSIONS" --sessions S8 S9 --output "$Kit\S8_S9_qc.csv"
