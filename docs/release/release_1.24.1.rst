GAMSPy 1.24.1 (2026-06-24)
==========================

Bug fixes
---------
- #630: Fix symbol.toDense bug when the records of the symbol is None.
- #670: Fix the behavior of monitor_process_tree_memory option. Setting this option True will cause GAMSPy to record the 
  high-memory mark for the GAMS process tree and print that information at the end of the script.
- #846: Add paranthesis to the representation of negative gp.Number objects since GAMS does not support expression like 3 + -7. It must be like 3 + (-7).
- #847: Fix missing gdx file issue in control loops.


