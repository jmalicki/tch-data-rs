# tch-data-rs

High-performance **Parquet → PyTorch** sequence data loaders, written in Rust with
[`tch-rs`](https://github.com/LaurentMazare/tch-rs) / [`pyo3-tch`](https://crates.io/crates/pyo3-tch)
and exposed to Python via [PyO3](https://pyo3.rs). Inspired by the (now-archived)
[torchdata](https://github.com/pytorch/data) pipeline style: composable readers,
multi-file round-robin prefetch, and batched device transfer.

> **Status:** experimental / research code. APIs target double-column Parquet sequences
> (e.g. LSTM training). Not yet published to crates.io or PyPI.

## Features

| Component | What it does |
| --- | --- |
| `ParquetToTorchSeqReaderFloat` | Read one Parquet file as overlapping `(x, y)` float tensors |
| `ParquetToTorchSeqRoundRobinFloat` | Round-robin across many files with background prefetch workers |
| `ParquetToTorchBatchedRoundRobinFloat` | Same as above, plus pad-to-batch + mask, with async host→device copy |

Sequences are formed by sliding windows over selected numeric columns. `forward_skips` controls
how many timesteps separate the input window `x` from the prediction target `y`.

## Requirements

- Rust **1.82+**
- [LibTorch](https://pytorch.org/get-started/locally/) / PyTorch **2.13.x** (matches `tch 0.26`)
- Python **3.9+** with `torch` installed (for the extension module)
- [maturin](https://www.maturin.rs/) to build/install the Python wheel

### Pointing Rust at LibTorch

On Apple Silicon, prefer a Python `torch` install (prebuilt libtorch wheels are not published for arm64):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install 'torch==2.13.0'
export LIBTORCH="$(python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')"
export DYLD_LIBRARY_PATH="${LIBTORCH}/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

On Linux x86_64 you can instead use Cargo's download helper:

```bash
cargo check --no-default-features --features download-libtorch
```

See also the [tch-rs README](https://github.com/LaurentMazare/tch-rs#getting-started).

## Quick start (Python)

```bash
# Install into the active virtualenv (enables extension-module)
maturin develop --features extension-module

python - <<'PY'
import tch_data

reader = tch_data.ParquetToTorchSeqReaderFloat(
    "data.parquet",
    colnames=["feature_a", "feature_b"],
    max_seq_len=128,
    forward_skips=1,
    augment_offset=False,
)
xy = reader.read_tensors(64)  # Optional[(x, y)]
print(None if xy is None else (xy[0].shape, xy[1].shape))

# Multi-file, batched
loader = tch_data.ParquetToTorchBatchedRoundRobinFloat(
    filenames=["shard0.parquet", "shard1.parquet"],
    colnames=["feature_a", "feature_b"],
    round_robin_size=2,   # parallel file workers
    buffer_size=8,        # prefetch depth per worker
    max_seq_len=128,
    forward_skips=1,
    augment_offset=False,
    batch_size=32,
)
for x, y, mask in loader:
    # x, y, mask are torch.Tensors (on CUDA when available)
    break
PY
```

### Python API summary

```text
ParquetToTorchSeqReaderFloat(filename, colnames, max_seq_len, forward_skips, augment_offset)
  .read_tensors(seq_len) -> Optional[Tuple[Tensor, Tensor]]

ParquetToTorchSeqRoundRobinFloat(filenames, colnames, round_robin_size, buffer_size,
                                 max_seq_len, forward_skips, augment_offset)
  # iterable of (x, y)

ParquetToTorchBatchedRoundRobinFloat(..., batch_size)
  # iterable of (x, y, mask)
```

## Rust usage

The crate builds as both `cdylib` (Python) and `rlib` (Rust). Core modules compile without
Python; enable the `python` feature for PyO3 bindings:

```bash
cargo check --no-default-features
cargo test  --no-default-features
cargo check --features python          # needs LIBTORCH + a Python interpreter
```

```rust
use tch_data::parquet_tensors::{
    ParquetLSTMSeqOptions, ParquetToTorchSeqReaderImpl,
};
use parquet::data_type::DoubleType;
use std::path::Path;

let mut reader = ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(
    Path::new("data.parquet"),
    &["feature_a", "feature_b"],
    ParquetLSTMSeqOptions {
        max_seq_len: 128,
        forward_skips: 1,
        augment_offset: false,
    },
)?;

while let Some((x, y)) = reader.get_tensor_len(64)? {
    // train step…
}
```

Public Rust modules:

- `tch_data::parquet_tensors` — single-file sequence reader
- `tch_data::round_robin` — multi-file prefetching iterator
- `tch_data::batcher` — pad / mask / device transfer

## Project layout

```text
src/
  lib.rs              Crate root (feature-gates Python)
  python.rs           PyO3 module (`tch_data`) via pyo3-tch
  parquet_tensors.rs  Parquet column → tensor sequences
  round_robin.rs      Multi-file worker pool + channel prefetch
  batcher.rs          Batching, padding, mask, device copy
```

## Building notes

| Goal | Command |
| --- | --- |
| Develop Python extension | `maturin develop --features extension-module` |
| Release wheel | `maturin build --release --features extension-module` |
| Rust typecheck (no Python) | `cargo check --no-default-features` |
| Format | `cargo fmt` |

### Dependency versions

| Crate | Version |
| --- | --- |
| `tch` / `pyo3-tch` | 0.26 |
| `pyo3` | 0.24 |
| `parquet` | 59 |
| `thiserror` | 2 |

## Migration from `parquet_lstm_tensor`

Earlier revisions used the crate / Python module name `parquet_lstm_tensor`. Imports should
now use `tch_data`:

```python
# before
import parquet_lstm_tensor as m

# after
import tch_data as m
```

Class names (`ParquetToTorchSeqReaderFloat`, etc.) are unchanged.

## License

MIT — see [LICENSE](LICENSE).
