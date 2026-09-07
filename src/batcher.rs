use tch::{kind::Kind, Device, Tensor};

use par_map::ParMap;

/// A padded mini-batch of variable-length sequences.
pub struct TensorBatchingItem {
    pub x: Tensor,
    pub y: Tensor,
    /// Boolean mask with the same shape as `x` / `y` before padding
    /// (ones for real timesteps, zeros for pad).
    pub mask: Tensor,
}

impl TensorBatchingItem {
    pub fn send_to_device(&self, device: Device, non_blocking: bool) -> TensorBatchingItem {
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

/// Prefers CUDA device 0 when available, otherwise CPU.
pub fn default_device() -> Device {
    if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

pub struct TensorBatchingIterator(Box<dyn Iterator<Item = TensorBatchingItem> + Send>);

impl TensorBatchingIterator {
    pub fn new(
        input: Box<dyn Iterator<Item = (Tensor, Tensor)> + Send>,
        batch_size: usize,
    ) -> Self {
        Self::new_with_device(input, batch_size, default_device())
    }

    pub fn new_with_device(
        input: Box<dyn Iterator<Item = (Tensor, Tensor)> + Send>,
        batch_size: usize,
        device: Device,
    ) -> Self {
        let iter = Box::new(
            input
                .pack(batch_size)
                .with_nb_threads(4)
                .par_map(move |batch| {
                    Self::batch(batch).send_to_device(device, device != Device::Cpu)
                }),
        );

        TensorBatchingIterator(iter)
    }

    /// Pad a list of `(x, y)` sequences into a single batch with a boolean mask.
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
