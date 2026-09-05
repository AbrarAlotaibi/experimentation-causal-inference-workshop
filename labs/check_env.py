"""Run this before Day 1: python check_env.py"""
import importlib, sys
required = ["numpy", "scipy", "pandas", "matplotlib", "statsmodels", "sklearn", "networkx"]
optional = ["dowhy", "econml", "linearmodels"]
print(f"Python {sys.version.split()[0]}")
ok = True
for m in required:
    try:
        importlib.import_module(m); print(f"  OK       {m}")
    except Exception:
        ok = False; print(f"  MISSING  {m}   -> pip install {m}")
for m in optional:
    try:
        importlib.import_module(m); print(f"  OK       {m} (optional)")
    except Exception:
        print(f"  optional {m} not installed (labs run without it)")
print("\nEnvironment ready." if ok else "\nInstall the missing packages above, then re-run.")
