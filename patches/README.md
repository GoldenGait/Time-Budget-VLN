# Patches

## `habitat_rearrange_fix.py`

Fixes a crash when importing `habitat_baselines` with `habitat-sim==0.2.3`.

The conda build of habitat-sim 0.2.3 does not include `habitat_sim.robots`, which
is required by the rearrangement task's action registration. Since VLN evaluation
does not use rearrangement, the fix wraps the problematic import in a try/except
so habitat loads cleanly.

**Apply to:** `habitat/tasks/rearrange/__init__.py` in your habitat-lab installation.

**Tested with:** habitat-sim 0.2.3 (source build), habitat-lab 0.2.1, Python 3.10.
