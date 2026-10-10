# Stock data

| File | What it is |
|---|---|
| `raw/indtrack1.txt` | Original OR-Library source file, unchanged |
| `hangseng_weekly.csv` | Same prices, one row per observation: `week,index,s01,...,s31` |
| `LICENSE_ORLIB.txt` | OR-Library MIT licence notice (verbatim) and attribution |

**Source.** J E Beasley, OR-Library, index tracking problem 1:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/indtrack1.txt
(the `http://` address redirects there). Downloaded 2026-10-10 during development, never at run time.
sha256 of `raw/indtrack1.txt`: `258b917e304e21c87ae481b9baba0aefe877b657fa63483255c23162a74aa629`.

**Observed format.** The format is the one documented on OR-Library's `indtrackinfo.html` page: line 1 is `N T`, then the
index values for t = 0..T, then each stock's values for t = 0..T.
- Header `31 290`, followed by 9,312 prices = 32 series (index + 31 stocks) x 291 observations.
- Layout (N+1)*(T+1). Every series starts on a new line (72 lines of 4 values, then 1 line of 3).
- The first index value is 8749.31759356. All prices are positive (stocks 1.79 to 484.59, index 8713.53 to 29550.18).

**What we do not know from the source.** The file has no dates and no stock names. `week` is
the observation number 0..290, and `s01..s31` follow the file order. The FRAPO R package documentation
(INDTRACK1, read 2026-10-10) describes the data as weekly prices of the Hang Seng index and 31 of its
constituents, March 1991 to September 1997, and says "The data was downloaded from DATASTREAM and has
been anonymized". We found no public list of the company names, and we do not guess them. We have not
re-checked the FRAPO description against Beasley, Meade & Chang (2003), the paper the OR-Library page cites.

Rebuild and validate: `python -m stocks.validate_data` (from the repo root).
