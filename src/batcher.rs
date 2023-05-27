
use tch::{TchError, Tensor, kind::Kind, Device};

struct TensorBatchingIterator {
    input: Box<dyn Iterator<(Tensor, Tensor)>>,
    batch_size: usize,
}

struct TensorBatchingItem {
    x: Tensor,
    y: Tensor,
    mask: Tensor,
}

impl Iterator for TensorBatchingIterator {
    type Item = TensorBatchingItem;

    fn next(&mut self) -> Option<Item> {
        let mut x_tensors = Vec::with_capacity(self.batch_size);
        let mut y_tensors = Vec::with_capacity(self.batch_size);
        let mut mask_tensors = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            let (x_tensor, y_tensor) = input.next();
            x_tensors.append(x_tensor);
            y_tensors.append(y_tensor);

            mask_tensors.append(
                Tensor.f_new_ones(
                    x_tensor.shape(),
                    Kind::Bool,
                    Device::Cpu,
                )
            )
        }
    }

    fn size_hint(self) -> (usize, Option<usize>) {
        let (lower, upper) = self.input.size_hint();

        let res_lower = lower / self.batch_size;
        let res_upper = upper.map(|high| (high + self.batch_size - 1) / batch_size);
        (res_lower, res_upper)
    }
}
