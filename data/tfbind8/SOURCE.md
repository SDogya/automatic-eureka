# TFBind8

`tfbind8.parquet` holds all 65,536 DNA strings of length 8 (lexicographic order, A < C < G < T)
with their TFBind8 binding score in [0, 1] (Design-Bench task `TFBind8-Exact-v0`, transcription factor SIX6).

Converted from `tfbind8-exact-v0-all.pkl` (keys `x`: int64 (65536, 8), `y`: float32 (65536, 1)), downloaded from
https://raw.githubusercontent.com/maxwshen/gflownet/main/datasets/tfbind8/tfbind8-exact-v0-all.pkl
(git blob sha1 `d51c6c21f11a663fef66fe8cff57d3984b1b5f54`, sha256 `eb9757a483467608d884d0bfebf0625f7a32c8576f566e02a5debfc02f99d9af`).
The pickle was read with an unpickler restricted to numpy arrays.
