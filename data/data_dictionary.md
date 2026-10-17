# Data dictionary

The source is a consolidated historical CSV supplied with the coursework. The original file was named `DWXX (3).csv`; the repository copy is `raw/dse_market_data.csv`. The source contains 1,614 rows from 2 January 2018 to 24 November 2024. One market date occurs twice; preprocessing retains the last row for that date in source-file order.

## Fields used by the analysis

| Field | Description |
|---|---|
| `Date` | Market observation date. |
| `DWX`, `DLX` | Damascus market indices as labeled in the source. |
| `ALAQ`, `NIC`, `SII`, `SKI`, `UIC`, `AROP` | DSE series as labeled in the source. `UIC` is the prediction target. |
| `gold`, `oil`, `SYP`, `^IRX` | External market and economic indicators as labeled in the source. |
| `TC`, `NC`, `TD`, `ND` | COVID-19 case and death indicators described in the coursework report. |

The source also contains `id`, `Change`, `Change%`, and pre-existing `pred*` fields. These are not model inputs. In particular, stored prediction columns are excluded to prevent training on values that may represent prior model outputs.

## Preprocessing

The pipeline parses dates, converts numeric strings (including thousands separators), sorts rows in ascending date order, and removes the repeated date while retaining its last source occurrence. It preserves zero values in the source indicators and does not infer that they are missing values. The final target row is omitted because it has no next-session close in the supplied snapshot.

The coursework report describes DSE, Yahoo Finance, and WHO as source providers for the respective market, external economic, and COVID-19 fields. The bundled CSV is a historical snapshot; it does not include row-level source URLs or retrieval timestamps.
