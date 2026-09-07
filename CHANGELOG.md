# Changelog

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
