# Changelog

## 0.2.0

Modernize dependencies and APIs for current `tch` / PyO3 / parquet.

### Changed
- `tch` **0.13 → 0.26** (PyTorch / LibTorch **2.13**)
- Tensor↔Python interop via **`pyo3-tch` 0.26** (replaces hand-rolled `PyTensor`)
- `pyo3` **0.18 → 0.24**
- `parquet` **40 → 59** (`read_batch` → `read_records`)
- `thiserror` **1 → 2**
- `pad_sequence` updated for the new `padding_side` argument
- Python bindings moved to `src/python.rs` behind the `python` feature
- Crate version **0.2.0**; MSRV **1.82**

### Added
- `download-libtorch` Cargo feature (Linux x86_64 convenience builds)
- Documented Apple Silicon LibTorch setup via a Python `torch` venv

## 0.1.0

Initial cleanup release of the former `parquet_lstm_tensor` prototype as **`tch-data`**.

### Added
- README, MIT license, `pyproject.toml` (maturin), rustfmt config, and GitHub Actions CI
- Proper crate metadata aligned with the `tch-data-rs` repository
- Both `cdylib` and `rlib` crate types (Python extension + Rust library)
- Device selection for batched transfers (`CUDA:0` when available, else CPU)

### Fixed
- Multi-row-group Parquet reads (`advance_row_groups` previously always panicked)
- Null handling when copying definition levels into tensors
- Error propagation from constructors (no more opaque `"Error creating class"`)
- Drop of round-robin workers no longer panics if the control channel is already closed

### Removed
- Unused `arrow` dependency (broke builds against newer `chrono`)
- Broken, unused `buffered_iterator` module

### Changed
- Python / crate module rename: `parquet_lstm_tensor` → `tch_data`
