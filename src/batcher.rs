

use std::cmp::max;
use std::num::TryFromIntError;

use tch::{Tensor, kind::Kind, Device};
use thiserror::Error;

struct TensorBatchingIterator {
    input: Box<dyn Iterator<Item = (Tensor, Tensor)>>,
    batch_size: usize,
}

pub struct TensorBatchingItem {
    x: Tensor,
    y: Tensor,
    mask: Tensor,
}

#[derive(Error, Debug)]
enum BatchingError {
    #[error("Empty batch")]
    EmptyBatch,
    #[error("Mismatched dimensions")]
    MismatchedDims,
    #[error(transparent)]
    TchError(
        #[from] tch::TchError
    ),
    #[error(transparent)]
    TryFromIntError(
        #[from] TryFromIntError
    ),
}

impl TensorBatchingIterator {
    // Make sure these differ only in first dimension
    // return size large enough for all
    fn check_dims(tensors: &[Tensor]) -> Result<Vec<i64>, BatchingError> {
        if tensors.is_empty() {
            return Err(BatchingError::EmptyBatch)
        }
        let mut shape: Vec<i64> = tensors[0].size().clone();

        for tensor in tensors[1..].iter() {
            let tensor_shape = tensor.size();
            if shape[1..] != tensor_shape[1..] {
                return Err(BatchingError::MismatchedDims)
            }

            shape[0] = max(shape[0], tensor_shape[0]);
        }

        return Ok(shape)
    }

}

impl Iterator for TensorBatchingIterator {
    type Item = TensorBatchingItem;

    fn next(&mut self) -> Option<Self::Item> {
        let mut x_tensors = Vec::with_capacity(self.batch_size);
        let mut y_tensors = Vec::with_capacity(self.batch_size);
        let mut mask_tensors = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            let (x_tensor, y_tensor) = match self.input.next() {
                Some((x_tensor, y_tensor)) => (x_tensor, y_tensor),
                None => {
                    if i == 0 { return None };
                    break
                }
            };
            let x_size = x_tensor.size();

            x_tensors.push(x_tensor);
            y_tensors.push(y_tensor);

            mask_tensors.push(
                Tensor::ones(
                    &x_size,
                    (Kind::Bool, Device::Cpu),
                )
            )
        }

        let x_shape = Self::check_dims(&x_tensors).expect("X Tensors have different shapes");
        let y_shape = Self::check_dims(&y_tensors).expect("Y Tensors have different shapes");

        if x_shape != y_shape {
            panic!(
                "X and Y tensors have differing shapes {x_shape:?} and {y_shape:?}",
            );
        }

        Some(TensorBatchingItem {
            x: Tensor::pad_sequence(&x_tensors, true, 0.0),
            y: Tensor::pad_sequence(&y_tensors, true, 0.0),
            mask: Tensor::pad_sequence(&mask_tensors, true, 0.0),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.input.size_hint();

        let res_lower = lower / self.batch_size;
        let res_upper = upper.map(|high| (high + self.batch_size - 1) / self.batch_size);
        (res_lower, res_upper)
    }
}
