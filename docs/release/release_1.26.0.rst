GAMSPy 1.26.0 (2026-07-06)
==========================

New features
------------
- #527: Add toDense function to implicit symbols so that indexed, sliced, transposed or permuted references can be converted to a dense array.

Bug fixes
---------
- #744: Fixes a bug where certain solve options (e.g. iteration_limit and time_limit) would incorrectly persist across subsequent model.solve() calls in case the subsequent solves do not redefine those options. These are now properly reset between solves unless explicitly specified.
- #850: Fix false negatives in domain validation in particular conditional expressions.
- #854: Revert breaking change in miro symbol column names.

Improved documentation
----------------------
- #821: Improve documentation SEO: add canonical URLs, Open Graph/Twitter card tags, an auto-generated ``sitemap.xml``, curated meta descriptions on key pages, and alt text for all images.

Miscellaneous internal changes
------------------------------
- #852: Allow nosolve licenses in GAMSPy.


