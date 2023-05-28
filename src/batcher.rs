
use tch::{TchError, Tensor, kind::Kind, Device, Shape};

struct TensorBatchingIterator {
    input: Box<dyn Iterator<(Tensor, Tensor)>>,
    batch_size: usize,
}

struct TensorBatchingItem {
    x: Tensor,
    y: Tensor,
    mask: Tensor,
}

impl TensorBatchingIterator {
    // Make sure these differ only in first dimension
    // return size large enough for all
    fn check_dims(tensors: &[Tensor]) -> Result<Shape> {
        if tensors.is_empty() {
            return Err()
        }
        let mut shape: Shape = tensors[0].shape;

        for tensor in tensors[1..] {
            if shape[1..] != tensor.shape[1..] {
                return Err()
            }

            shape[0] = max(shape[0], tensor.shape[0]);
        }

        return shape
    }

    fn batch_tensors_with_shape(shape: &[i64], tensors: &[Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            panic!("Cannot batch empty tensors");
        }
        let mut out_tensor = Tensor.f_new_zeros(
            &out_shape,
            tensors[0].kind(),
            tensors[0].device(),
        ).expect("Tensor creation failed");

        for i in 0..tensors.len() {
            let tensor = &tensors[i];
            let mut item_view = out_tensor.f_slice(0, Some(i), Some(i+1), 1);
            let mut window_view = item_view.f_slice(1, Some(0), Some(tensor.shape[0]));

            let t = window_view.fill_tensor_(tensor);
            if !t.equals(window_view) {
                panic!("Did not mutate tensor!");
            }
        }

        Ok(out_tensor)
    }

    fn batch_tensors(tensors: &[Tensor]) -> Result<Tensor> {
        let shape = Self::check_dims(tensors)?;
        batch_tensors_with_shape(shape, tensors)
    }
}

impl Iterator for TensorBatchingIterator {
    type Item = TensorBatchingItem;

    fn next(&mut self) -> Option<Item> {
        let mut x_tensors = Vec::with_capacity(self.batch_size);
        let mut y_tensors = Vec::with_capacity(self.batch_size);
        let mut mask_tensors = Vec::with_capacity(self.batch_size);

        for i in 0..self.batch_size {
            let (x_tensor, y_tensor) = match input.next() {
                Some(x_tensor, y_tensor) => (x_tensor, y_tensor),
                None => {
                    if i == 0 { return None };
                    break
                }
            }
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

        let x_shape = self.check_dims(x_tensors).expect("X Tensors have different shapes");
        let y_shape = self.check_dims(y_tensors).expect("Y Tensors have different shapes");

        if x_shape != y_shape { panic!("X and Y tensors have differing shapes {x_shape} and {y_shape}"); }
        let mut out_shape: Vec<i64> = vec![x_tensors.len()];
        out_shape.extend_from_slice(&x_shape);

        Some(TensorBatchingItem {
            x: self.batch_tensors_with_shape(x_shape, x_tensors).expect("Batching x tensors"),
            y: self.batch_tensors_with_shape(x_shape, y_tensors).expect("Batching y tensors"),
            mask: self.batch_tensors_with_shape(x_shape, mask_tensors).expect("Batching mask tensors"),
        })
    }

    fn size_hint(self) -> (usize, Option<usize>) {
        let (lower, upper) = self.input.size_hint();

        let res_lower = lower / self.batch_size;
        let res_upper = upper.map(|high| (high + self.batch_size - 1) / batch_size);
        (res_lower, res_upper)
    }
}


#[cfg(test)]
mod test {

    #[test]
    fn test_batch_one() {
        let shape = [3, 5, 7];
        let tensors = [Tensor.random(shape)?];
        let batched_tensor = TensorBatchingIterator::batch_tensors(&tensors)?;

        assert_eq!([1, 3, 5, 7], batched_tensor.shape);

    }
}