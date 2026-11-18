# Install and check

Use a separate checkout. Preserve the original Windows working project and its
dated backup; overlaying the new tree would leave old and moved copies together.

```powershell
git clone --branch research/session-robustness-generator https://github.com/tristsou/TriCSI.git TriCSI-clean
cd TriCSI-clean
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tricsi doctor --code-only
```

Python 3.12 is the software-test target, not a verified historical requirement.
For exact fitted scikit-learn/PyTorch models, restore the original environment
using the release's Python version, requirements lock and environment records.
The root requirements are a starter dependency list, not a historical lock.
The owner supplied an actual [working-environment lock](../metadata/owner_requirements_lock.txt)
and [Windows/Python details](../metadata/owner_environment.json). That captured
environment uses Python 3.14.6 and CPU PyTorch 2.12.1. On matching Windows/Python,
install its lock with `python -m pip install -r metadata/owner_requirements_lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`.
Do not assume that this captured environment identifies every older training run.
For CPU PyTorch see https://pytorch.org/get-started/locally/ .

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/session_robustness/tests -v
python -m unittest discover -s experiments/session_diagnostics/tests -v
```

Before real evaluation run `python -m tricsi doctor`. Missing assets on a code-only
clone are expected. This checks source hashes and file existence, not physical
labels or historical checkpoint identity; verify the release and run manifests.
