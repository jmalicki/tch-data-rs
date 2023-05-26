
use pyo3::prelude::*;
//use pyo3::PyIterProtocol;
use pyo3::{AsPyPointer, exceptions::{PyException, PyTypeError, PyValueError}};

use parquet::data_type::DoubleType;
mod parquet_tensors;
use parquet_tensors::{ParquetToTorchSeqReaderImpl, ParquetLSTMTensorError};

struct PyTensor(tch::Tensor);


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

#[pyclass(module="parquet_lstm_tensor")]
struct ParquetToTorchSeqReaderFloat {
    reader: ParquetToTorchSeqReaderImpl<f32, DoubleType>,
}

#[pymethods]
impl ParquetToTorchSeqReaderFloat {

    #[new]
    fn __new__(filename: &str, colnames: Vec::<String>, max_seq_len: usize, forward_skips: usize, augment_offset: bool) -> PyResult<Self> {
        let path = std::path::Path::new(filename);
        match ParquetToTorchSeqReaderImpl::<f32, DoubleType>::new(path, &colnames, max_seq_len, forward_skips, augment_offset) {
            Ok(reader) => Ok(ParquetToTorchSeqReaderFloat { reader }),
            Err(_e) => Err(PyErr::new::<PyException, _>("Error creating class")),
        }
    }

    fn read_tensors(&mut self, seq_len: i32) -> PyResult<Option<(PyTensor, PyTensor)>> {
        match self.reader.get_tensor_len(seq_len.try_into()?)? {
            Some((tensor_x, tensor_y)) => {
                Ok(Some(
                    (PyTensor(tensor_x), PyTensor(tensor_y))
                ))
            },
            None => Ok(None)
        }
    }
}


// #[pyproto]
// impl PyIterProtocol for ParquetToTorchSeqReaderFloat {
//     fn __next__(mut slf: PyRefMut<Self>) -> IterNextOutput<usize, &'static str> {
//         if slf.count < 5 {
//             slf.count += 1;
//             IterNextOutput::Yield(slf.count)
//         } else {
//             IterNextOutput::Return("Ended")
//         }
//     }
// }


#[pymodule]
fn parquet_lstm_tensor(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    _py.import("torch")?;
    m.add_class::<ParquetToTorchSeqReaderFloat>()?;
    Ok(())
}
