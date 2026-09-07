//! Python bindings for Parquet → PyTorch sequence loaders.
//!
//! Import as `import tch_data` after building the extension with maturin.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::batcher::TensorBatchingIterator;
use crate::parquet_tensors::{
    ParquetLSTMSeqOptions, ParquetLSTMTensorError, ParquetToTorchSeqReaderFactory,
    ParquetToTorchSeqReaderImpl,
};
use crate::round_robin::{RoundRobin, RoundRobinIterator, RoundRobinOptions};
use parquet::data_type::DoubleType;
use tch::Tensor;

impl From<ParquetLSTMTensorError> for pyo3::PyErr {
    fn from(value: ParquetLSTMTensorError) -> Self {
        use ParquetLSTMTensorError::*;

        match value {
            IoError(e) => e.into(),
            TryFromIntError(e) => e.into(),
            Tensor(e) => wrap_tch_err(e),
            other => PyErr::new::<PyException, _>(format!("{other}")),
        }
    }
}

#[pyclass(module = "tch_data", unsendable)]
struct ParquetToTorchSeqReaderFloat {
    reader: ParquetToTorchSeqReaderImpl<f32, DoubleType>,
}

#[pyclass(module = "tch_data")]
struct ParquetToTorchSeqRoundRobinFloat {
    reader: RoundRobin<(Tensor, Tensor)>,
}

#[pyclass(module = "tch_data")]
struct ParquetToTorchSeqRoundRobinFloatIter {
    iter: Mutex<RoundRobinIterator<(Tensor, Tensor)>>,
}

#[pyclass(module = "tch_data")]
struct ParquetToTorchBatchedRoundRobinFloat {
    round_robin: RoundRobin<(Tensor, Tensor)>,
    batch_size: usize,
}

#[pyclass(module = "tch_data")]
struct ParquetToTorchBatchedRoundRobinFloatIter {
    iter: Mutex<TensorBatchingIterator>,
}

fn make_round_robin(
    filenames: &[String],
    colnames: Vec<String>,
    round_robin_size: usize,
    buffer_size: usize,
    max_seq_len: usize,
    forward_skips: usize,
    augment_offset: bool,
) -> PyResult<RoundRobin<(Tensor, Tensor)>> {
    let reader_create: Arc<ParquetToTorchSeqReaderFactory<(Tensor, Tensor)>> =
        Arc::new(move |path: &Path| {
            Ok(Box::new(
                ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(
                    path,
                    &colnames,
                    ParquetLSTMSeqOptions {
                        max_seq_len,
                        forward_skips,
                        augment_offset,
                    },
                )?,
            ))
        });

    RoundRobin::<(Tensor, Tensor)>::new(
        filenames,
        RoundRobinOptions {
            round_robin_size,
            buffer_size,
        },
        reader_create,
    )
    .map_err(|e| PyErr::new::<PyException, _>(format!("failed to create round-robin reader: {e}")))
}

#[pymethods]
impl ParquetToTorchBatchedRoundRobinFloat {
    #[new]
    fn __new__(
        filenames: Vec<String>,
        colnames: Vec<String>,
        round_robin_size: usize,
        buffer_size: usize,
        max_seq_len: usize,
        forward_skips: usize,
        augment_offset: bool,
        batch_size: usize,
    ) -> PyResult<Self> {
        let round_robin = make_round_robin(
            &filenames,
            colnames,
            round_robin_size,
            buffer_size,
            max_seq_len,
            forward_skips,
            augment_offset,
        )?;

        Ok(ParquetToTorchBatchedRoundRobinFloat {
            round_robin,
            batch_size,
        })
    }

    fn __iter__(&self) -> ParquetToTorchBatchedRoundRobinFloatIter {
        let batcher =
            TensorBatchingIterator::new(Box::new(self.round_robin.iter()), self.batch_size);

        ParquetToTorchBatchedRoundRobinFloatIter {
            iter: Mutex::new(batcher),
        }
    }
}

#[pymethods]
impl ParquetToTorchBatchedRoundRobinFloatIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<(PyTensor, PyTensor, PyTensor)>> {
        let item = py.allow_threads(|| {
            let mut iter = self.iter.lock().expect("lock failed");
            iter.next()
        });
        Ok(item.map(|batching_item| {
            (
                PyTensor(batching_item.x),
                PyTensor(batching_item.y),
                PyTensor(batching_item.mask),
            )
        }))
    }
}

#[pymethods]
impl ParquetToTorchSeqReaderFloat {
    #[new]
    fn __new__(
        filename: &str,
        colnames: Vec<String>,
        max_seq_len: usize,
        forward_skips: usize,
        augment_offset: bool,
    ) -> PyResult<Self> {
        let path = Path::new(filename);
        let reader = ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(
            path,
            &colnames,
            ParquetLSTMSeqOptions {
                max_seq_len,
                forward_skips,
                augment_offset,
            },
        )?;
        Ok(ParquetToTorchSeqReaderFloat { reader })
    }

    fn read_tensors(
        &mut self,
        py: Python<'_>,
        seq_len: i32,
    ) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let seq_len: usize = seq_len.try_into()?;
        let res = py.allow_threads(|| self.reader.get_tensor_len(seq_len))?;
        Ok(res.map(|(tensor_x, tensor_y)| (PyTensor(tensor_x), PyTensor(tensor_y))))
    }
}

#[pymethods]
impl ParquetToTorchSeqRoundRobinFloat {
    #[new]
    fn __new__(
        filenames: Vec<String>,
        colnames: Vec<String>,
        round_robin_size: usize,
        buffer_size: usize,
        max_seq_len: usize,
        forward_skips: usize,
        augment_offset: bool,
    ) -> PyResult<Self> {
        let reader = make_round_robin(
            &filenames,
            colnames,
            round_robin_size,
            buffer_size,
            max_seq_len,
            forward_skips,
            augment_offset,
        )?;
        Ok(ParquetToTorchSeqRoundRobinFloat { reader })
    }

    fn __iter__(&self) -> PyResult<ParquetToTorchSeqRoundRobinFloatIter> {
        Ok(ParquetToTorchSeqRoundRobinFloatIter {
            iter: Mutex::new(self.reader.iter()),
        })
    }
}

#[pymethods]
impl ParquetToTorchSeqRoundRobinFloatIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let item = py.allow_threads(|| {
            let mut iter = self.iter.lock().expect("lock failed");
            iter.next()
        });
        Ok(item.map(|(tensor_x, tensor_y)| (PyTensor(tensor_x), PyTensor(tensor_y))))
    }
}

#[pymodule]
fn tch_data(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py.import("torch")?;
    m.add_class::<ParquetToTorchSeqReaderFloat>()?;
    m.add_class::<ParquetToTorchSeqRoundRobinFloat>()?;
    m.add_class::<ParquetToTorchBatchedRoundRobinFloat>()?;
    Ok(())
}
