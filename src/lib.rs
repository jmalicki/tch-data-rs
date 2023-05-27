use pyo3::prelude::*;
use pyo3::pyclass::IterNextOutput;
use pyo3::{
    exceptions::{PyException, PyTypeError, PyValueError},
    AsPyPointer,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

use parquet::data_type::DoubleType;
mod parquet_tensors;
mod round_robin;
use tch::Tensor;

use parquet_tensors::{
    ParquetLSTMTensorError, ParquetToTorchSeqReaderFactory, ParquetToTorchSeqReaderImpl,
};
use round_robin::{RoundRobin, RoundRobinIterator};

struct PyTensor(Tensor);

impl From<ParquetLSTMTensorError> for pyo3::PyErr {
    fn from(value: ParquetLSTMTensorError) -> Self {
        use ParquetLSTMTensorError::*;

        match value {
            IoError(e) => e.into(),
            TryFromIntError(e) => e.into(),
            Tensor(e) => wrap_tch_err(e),
            other => PyErr::new::<PyException, _>(format!("{}", other)),
        }
    }
}

fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(PyTensor)
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}

#[pyclass(module = "parquet_lstm_tensor")]
struct ParquetToTorchSeqReaderFloat {
    reader: ParquetToTorchSeqReaderImpl<f32, DoubleType>,
}

#[pyclass(module = "parquet_lstm_tensor")]
struct ParquetToTorchSeqRoundRobinFloat {
    reader: RoundRobin<(Tensor, Tensor)>,
}

#[pyclass(module = "parquet_lstm_tensor")]
struct ParquetToTorchSeqRoundRobinFloatIter {
    iter: Mutex<RoundRobinIterator<(Tensor, Tensor)>>,
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
        let path = std::path::Path::new(filename);
        match ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(
            path,
            &colnames,
            max_seq_len,
            forward_skips,
            augment_offset,
        ) {
            Ok(reader) => Ok(ParquetToTorchSeqReaderFloat { reader }),
            Err(_e) => Err(PyErr::new::<PyException, _>("Error creating class")),
        }
    }

    fn read_tensors(
        &mut self,
        py: Python<'_>,
        seq_len: i32,
    ) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let res = py.allow_threads(move || self.reader.get_tensor_len(seq_len.try_into()?))?;
        match res {
            Some((tensor_x, tensor_y)) => Ok(Some((PyTensor(tensor_x), PyTensor(tensor_y)))),
            None => Ok(None),
        }
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
        let reader_create: Arc<ParquetToTorchSeqReaderFactory<(Tensor, Tensor)>> =
            Arc::new(move |path: &Path| {
                Ok(Box::new(
                    ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(
                        path,
                        &colnames,
                        max_seq_len,
                        forward_skips,
                        augment_offset,
                    )?,
                ))
            });

        match RoundRobin::<(Tensor, Tensor)>::new(
            &filenames,
            round_robin_size,
            buffer_size,
            reader_create,
        ) {
            Ok(reader) => Ok(ParquetToTorchSeqRoundRobinFloat { reader }),
            Err(_e) => Err(PyErr::new::<PyException, _>("Error creating class")),
        }
    }

    fn __iter__(&self) -> PyResult<ParquetToTorchSeqRoundRobinFloatIter> {
        Ok(ParquetToTorchSeqRoundRobinFloatIter {
            iter: Mutex::new(self.reader.into_iter()),
        })
    }
}

#[pymethods]
impl ParquetToTorchSeqRoundRobinFloatIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<IterNextOutput<(PyTensor, PyTensor), &'static str>> {
        match py.allow_threads(move || {
            let mut iter = self.iter.lock().expect("lock failed");
            iter.next()
        }) {
            None => Ok(IterNextOutput::Return("Ended")),
            Some((tensor_x, tensor_y)) => Ok(IterNextOutput::Yield((
                PyTensor(tensor_x),
                PyTensor(tensor_y),
            ))),
        }
    }
}

#[pymodule]
fn parquet_lstm_tensor(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    py.import("torch")?;
    m.add_class::<ParquetToTorchSeqReaderFloat>()?;
    m.add_class::<ParquetToTorchSeqRoundRobinFloat>()?;
    Ok(())
}
