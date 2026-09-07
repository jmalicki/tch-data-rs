# tch-data-rs

High-performance **Parquet → PyTorch** sequence data loaders, written in Rust with
[`tch-rs`](https://github.com/LaurentMazare/tch-rs) and exposed to Python via
[PyO3](https://pyo3.rs). Inspired by the (now-archived) [torchdata](https://github.com/pytorch/data)
pipeline style: composable readers, multi-file round-robin prefetch, and batched CUDA transfer.

> **Status:** experimental / research code. The APIs work for double-column Parquet sequences
> (e.g. LSTM training), but dependency versions are pinned to the `tch 0.13` / `pyo3 0.18` era
> and the project is not yet published to crates.io or PyPI.

## Features

| Component | What it does |
| --- | --- |
| `ParquetToTorchSeqReaderFloat` | Read one Parquet file as overlapping `(x, y)` float tensors |
| `ParquetToTorchSeqRoundRobinFloat` | Round-robin across many files with background prefetch workers |
| `ParquetToTorchBatchedRoundRobinFloat` | Same as above, plus pad-to-batch + mask, with async host→device copy |

Sequences are formed by sliding windows over selected numeric columns. `forward_skips` controls
how many timesteps separate the input window `x` from the prediction target `y`.

## Requirements

- Rust 1.70+
- [LibTorch](https://pytorch.org/get-started/locally/) matching `tch 0.13` (typically PyTorch 2.0.x)
- Python 3.8+ with `torch` installed (for the extension module)
- [maturin](https://www.maturin.rs/) to build/install the Python wheel

Set `LIBTORCH` (and on macOS often `LIBTORCH_INCLUDE` / `DYLD_LIBRARY_PATH`) as described in the
[tch-rs README](https://github.com/LaurentMazare/tch-rs#getting-started).

## Quick start (Python)

```bash
# Install into the active virtualenv
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

The crate builds as both `cdylib` (Python) and `rlib` (Rust). For Rust-only checks/tests,
disable the default feature so the linker does not require `libpython`:

```bash
cargo check --no-default-features
cargo test  --no-default-features
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
  lib.rs              Python module (`tch_data`) + PyO3 wrappers
  parquet_tensors.rs  Parquet column → tensor sequences
  round_robin.rs      Multi-file worker pool + channel prefetch
  batcher.rs          Batching, padding, mask, device copy
```

## Building notes

| Goal | Command |
| --- | --- |
| Develop Python extension | `maturin develop` |
| Release wheel | `maturin build --release` |
| Rust typecheck (no Python link) | `cargo check --no-default-features` |
| Format | `cargo fmt` |

`parquet` is pinned at **40.x** to stay compatible with the `tch 0.13` / older Arrow stack.
The unused `arrow` dependency from earlier revisions was removed.

## Migration from `parquet_lstm_tensor`

Earlier revisions of this repository used the crate / Python module name
`parquet_lstm_tensor`. Imports should now use `tch_data`:

```python
# before
import parquet_lstm_tensor as m

# after
import tch_data as m
```

Class names (`ParquetToTorchSeqReaderFloat`, etc.) are unchanged.

## License

MIT — see [LICENSE](LICENSE).
