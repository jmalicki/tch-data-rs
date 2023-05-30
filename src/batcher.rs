use std::num::TryFromIntError;
use std::{cmp::max, sync::Arc};

use tch::{kind::Kind, Device, Tensor};
use thiserror::Error;

use par_map::ParMap;

pub struct TensorBatchingItem {
    pub x: Tensor,
    pub y: Tensor,
    pub mask: Tensor,
}

impl TensorBatchingItem {
    pub fn send_to_device(&self, device: tch::Device, non_blocking: bool) -> TensorBatchingItem {
        TensorBatchingItem {
            x: self
                .x
                .pin_memory(device)
                .to_device_(device, self.x.kind(), non_blocking, false),
            y: self
                .y
                .pin_memory(device)
                .to_device_(device, self.y.kind(), non_blocking, false),
            mask: self.mask.pin_memory(device).to_device_(
                device,
                self.mask.kind(),
                non_blocking,
                false,
            ),
        }
    }
}

pub struct TensorBatchingIterator(Box<dyn Iterator<Item = TensorBatchingItem> + Send>);

#[derive(Error, Debug)]
enum BatchingError {
    #[error("Empty batch")]
    EmptyBatch,
    #[error("Mismatched dimensions")]
    MismatchedDims,
    #[error(transparent)]
    TchError(#[from] tch::TchError),
    #[error(transparent)]
    TryFromIntError(#[from] TryFromIntError),
}

impl TensorBatchingIterator {
    pub fn new(
        input: Box<dyn Iterator<Item = (Tensor, Tensor)> + Send>,
        batch_size: usize,
    ) -> Self {
        let iter = Box::new(
            input
                .pack(batch_size)
                .with_nb_threads(4)
                .par_map(|batch| Self::batch(batch).send_to_device(tch::Device::Cuda(0), true)),
        );

        TensorBatchingIterator(iter)
    }

    pub fn batch(batch: Vec<(Tensor, Tensor)>) -> TensorBatchingItem {
        let mut x_tensors = Vec::with_capacity(batch.len());
        let mut y_tensors = Vec::with_capacity(batch.len());
        let mut mask_tensors = Vec::with_capacity(batch.len());

        for (x, y) in batch {
            let x_size = x.size();

            x_tensors.push(x);
            y_tensors.push(y);
            mask_tensors.push(Tensor::ones(&x_size, (Kind::Bool, Device::Cpu)));
        }

        TensorBatchingItem {
            x: Tensor::pad_sequence(&x_tensors, true, 0.0),
            y: Tensor::pad_sequence(&y_tensors, true, 0.0),
            mask: Tensor::pad_sequence(&mask_tensors, true, 0.0),
        }
    }
}

impl Iterator for TensorBatchingIterator {
    type Item = TensorBatchingItem;

    fn next(&mut self) -> Option<TensorBatchingItem> {
        self.0.next()
    }
}
