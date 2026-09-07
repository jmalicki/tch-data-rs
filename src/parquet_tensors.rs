use core::hash::Hash;
use num::{NumCast, ToPrimitive};
use parquet::data_type::DataType;
use parquet::{
    column::reader::ColumnReaderImpl,
    file::{
        metadata::ParquetMetaData,
        reader::{FileReader, RowGroupReader, SerializedFileReader},
    },
};
use std::cmp::Eq;
use std::collections::HashMap;
use std::{fs::File, path::Path};
use tch::{TchError, Tensor};
use thiserror::Error;

/// Per-row-group cursor over typed column readers.
struct ParquetToTorchSeqReaderRowGroup<ParquetValType>
where
    ParquetValType: DataType,
{
    col_readers: Vec<Option<ColumnReaderImpl<ParquetValType>>>,
    row_group_num: usize,
    row_group_offset: usize,
    row_group_num_rows: usize,
}

pub type ParquetToTorchSeqReaderFactory<Item> =
    dyn Fn(&Path) -> Result<Box<dyn Iterator<Item = Item>>> + Send + Sync;

pub trait ParquetToTorchSeqReader {
    fn get_tensors(&mut self) -> Result<Option<(Tensor, Tensor)>>;
    fn eof(&mut self) -> Result<bool>;
}

/// Options controlling how sequences are sliced from a Parquet file.
pub struct ParquetLSTMSeqOptions {
    pub max_seq_len: usize,
    /// Number of timesteps between input `x` and prediction target `y`.
    pub forward_skips: usize,
    /// Reserved: when `true`, randomly offset the first window within a file.
    /// Not yet implemented — accepted for API stability with the Python bindings.
    pub augment_offset: bool,
}

pub struct ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
where
    ParquetValType: DataType,
{
    reader: SerializedFileReader<File>,
    col_indices: Vec<usize>,
    max_levels: Vec<i16>,
    num_row_groups: usize,
    iter: ParquetToTorchSeqReaderRowGroup<ParquetValType>,
    last_rows: Vec<TensorValType>,
    seq_options: ParquetLSTMSeqOptions,
}

impl<ParquetValType: DataType> ParquetToTorchSeqReaderRowGroup<ParquetValType> {
    fn new(row_group_num: usize, num_cols: usize) -> Self {
        ParquetToTorchSeqReaderRowGroup {
            row_group_num,
            row_group_num_rows: 0,
            row_group_offset: 0,
            col_readers: {
                let mut v = Vec::new();
                v.resize_with(num_cols, || None);
                v
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum ParquetLSTMTensorError {
    #[error("Could not find column {column} in parquet metadata")]
    ColumnNotFoundError { column: String },
    #[error("Short read of column")]
    ShortReadError,

    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),

    #[error(transparent)]
    Tensor(#[from] TchError),

    #[error(transparent)]
    TryFromIntError(#[from] std::num::TryFromIntError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = core::result::Result<T, ParquetLSTMTensorError>;

pub trait TensorValTypeTrait: tch::kind::Element + Default + NumCast {}
impl<T> TensorValTypeTrait for T where T: tch::kind::Element + Default + NumCast {}

pub trait ParquetValTypeTrait: DataType {}
impl<ValType> ParquetValTypeTrait for ValType
where
    ValType: DataType,
    ValType::T: Copy + Default + NumCast + ToPrimitive,
{
}

impl<TensorValType: TensorValTypeTrait, ParquetValType: ParquetValTypeTrait>
    ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
where
    ParquetValType::T: NumCast + ToPrimitive,
{
    /// Return indices and max definition levels for `colnames` in schema order.
    fn column_indices_and_levels<StrRef: AsRef<str> + Hash + Eq>(
        metadata: &ParquetMetaData,
        colnames: &[StrRef],
    ) -> Result<(Vec<usize>, Vec<i16>)> {
        let mut colmap = colnames
            .iter()
            .enumerate()
            .map(|(idx, val)| (val.as_ref(), idx))
            .collect::<HashMap<_, _>>();
        let mut indices: Vec<usize> = vec![0; colnames.len()];
        let mut max_levels: Vec<i16> = vec![0; colnames.len()];

        for (i, col) in metadata
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .enumerate()
        {
            if let Some(idx) = colmap.remove(col.name()) {
                indices[idx] = i;
                max_levels[idx] = col.max_def_level();
            }
        }

        match colmap.keys().next() {
            Some(column) => Err(ParquetLSTMTensorError::ColumnNotFoundError {
                column: column.to_string(),
            }),
            None => Ok((indices, max_levels)),
        }
    }

    pub fn new<StrRef: AsRef<str> + Hash + Eq>(
        filename: &Path,
        colnames: &[StrRef],
        seq_options: ParquetLSTMSeqOptions,
    ) -> Result<Self> {
        let f = File::open(filename)?;
        let num_cols = colnames.len();

        let reader = SerializedFileReader::new(f)?;
        let parquet_metadata = reader.metadata();

        let (col_indices, max_levels) =
            Self::column_indices_and_levels(parquet_metadata, colnames)?;
        let num_row_groups = parquet_metadata.num_row_groups();

        let iter_info = Self::next_row_group(&reader, 0, num_row_groups, &col_indices)?;

        let forward_skips = seq_options.forward_skips;
        let mut res = ParquetToTorchSeqReaderImpl {
            reader,
            num_row_groups,
            col_indices,
            max_levels,
            iter: iter_info,
            last_rows: Vec::new(),
            seq_options,
        };

        let mut last_rows = vec![TensorValType::default(); forward_skips * num_cols];

        let num_read = res.read_data(&mut last_rows, 0, forward_skips)?;
        res.last_rows = last_rows;

        if num_read < forward_skips && !res.eof()? {
            return Err(ParquetLSTMTensorError::ShortReadError);
        }

        Ok(res)
    }

    fn init_col_readers(
        row_group_reader: &dyn RowGroupReader,
        col_indices: &[usize],
    ) -> Result<Vec<Option<ColumnReaderImpl<ParquetValType>>>> {
        let readers = col_indices
            .iter()
            .copied()
            .map(|idx| {
                row_group_reader.get_column_reader(idx).map(|col_reader| {
                    Some(parquet::column::reader::get_typed_column_reader::<
                        ParquetValType,
                    >(col_reader))
                })
            })
            .collect::<core::result::Result<Vec<_>, _>>()?;
        Ok(readers)
    }

    fn next_row_group(
        reader: &dyn FileReader,
        row_group_num: usize,
        num_row_groups: usize,
        col_indices: &[usize],
    ) -> Result<ParquetToTorchSeqReaderRowGroup<ParquetValType>> {
        if row_group_num >= num_row_groups {
            Ok(ParquetToTorchSeqReaderRowGroup::new(
                row_group_num,
                col_indices.len(),
            ))
        } else {
            let row_group = reader.get_row_group(row_group_num)?;
            let row_group_num_rows = row_group.metadata().num_rows().try_into()?;
            let col_readers = Self::init_col_readers(&*row_group, col_indices)?;

            Ok(ParquetToTorchSeqReaderRowGroup {
                row_group_num,
                row_group_num_rows,
                row_group_offset: 0,
                col_readers,
            })
        }
    }

    fn advance_row_groups(&mut self) -> Result<()> {
        while self.iter.row_group_offset == self.iter.row_group_num_rows {
            // Already past the last group (empty sentinel), or no groups at all.
            if self.iter.row_group_num >= self.num_row_groups {
                return Ok(());
            }

            let next = self.iter.row_group_num + 1;
            self.iter =
                Self::next_row_group(&self.reader, next, self.num_row_groups, &self.col_indices)?;

            // next_row_group returns an empty sentinel when `next >= num_row_groups`.
            if next >= self.num_row_groups {
                return Ok(());
            }
        }

        Ok(())
    }

    fn read_row_group_data(
        &mut self,
        data: &mut [TensorValType],
        cur_offset: usize,
        seq_len: usize,
    ) -> Result<usize> {
        let num_cols: usize = self.col_indices.len();

        if seq_len * num_cols > data.len() {
            panic!(
                "Data requested {data_req} larger than buffer {data_len}",
                data_req = seq_len * num_cols,
                data_len = data.len(),
            )
        }

        self.advance_row_groups()?;
        if cur_offset > seq_len {
            panic!("offset {cur_offset} greater than sequence length {seq_len}")
        }

        let num_rows: usize = core::cmp::min(
            seq_len - cur_offset,
            self.iter.row_group_num_rows - self.iter.row_group_offset,
        );

        if num_rows == 0 {
            return Ok(0);
        }

        let mut buffer: Vec<ParquetValType::T> = Vec::with_capacity(num_rows);
        let mut def_levels: Vec<i16> = Vec::with_capacity(num_rows);
        let mut rep_levels: Vec<i16> = Vec::with_capacity(num_rows);

        for (col_idx, col_reader) in self.iter.col_readers.iter_mut().enumerate() {
            if let Some(reader) = col_reader.as_mut() {
                buffer.clear();
                def_levels.clear();
                rep_levels.clear();

                let (_records_read, non_null_read, levels_read) = reader.read_records(
                    num_rows,
                    Some(&mut def_levels),
                    Some(&mut rep_levels),
                    &mut buffer,
                )?;
                if levels_read != num_rows {
                    return Err(ParquetLSTMTensorError::ShortReadError);
                }
                if non_null_read > levels_read {
                    panic!("Expected non nulls ({non_null_read}) to be <= levels ({levels_read})")
                }

                let max_level = self.max_levels[col_idx];
                if max_level == 0 {
                    // Required column: every level is a value.
                    for dest_idx in 0..levels_read {
                        data[(cur_offset + dest_idx) * num_cols + col_idx] =
                            num::cast(buffer[dest_idx].clone()).unwrap();
                    }
                } else {
                    // def_levels is indexed by row position; buffer only holds non-nulls.
                    let mut src_idx: usize = 0;
                    for dest_idx in 0..levels_read {
                        if def_levels[dest_idx] == max_level {
                            if src_idx >= non_null_read {
                                panic!(
                                    "More non-null values ({src_idx}) than reported read {non_null_read}!"
                                )
                            }
                            data[(cur_offset + dest_idx) * num_cols + col_idx] =
                                num::cast(buffer[src_idx].clone()).unwrap();
                            src_idx += 1;
                        }
                    }
                }
            }
        }

        self.iter.row_group_offset += num_rows;

        Ok(num_rows)
    }

    pub fn read_data(
        &mut self,
        data: &mut [TensorValType],
        cur_offset: usize,
        seq_len: usize,
    ) -> Result<usize> {
        let mut rows_read = 0;

        while cur_offset + rows_read < seq_len {
            let num_rows = self.read_row_group_data(data, cur_offset + rows_read, seq_len)?;
            rows_read += num_rows;
            if num_rows == 0 {
                break;
            }
        }

        Ok(rows_read)
    }

    pub fn get_tensor_len(&mut self, seq_len: usize) -> Result<Option<(Tensor, Tensor)>> {
        let num_cols: usize = self.col_indices.len();
        let mut data: Vec<TensorValType> =
            vec![TensorValType::default(); (self.seq_options.forward_skips + seq_len) * num_cols];

        data[..(num_cols * self.seq_options.forward_skips)].clone_from_slice(&self.last_rows);

        let mut cur_offset: usize = self.seq_options.forward_skips;

        let rows_read = self.read_data(
            &mut data,
            cur_offset,
            seq_len + self.seq_options.forward_skips,
        )?;
        cur_offset += rows_read;

        if cur_offset == self.seq_options.forward_skips {
            return Ok(None);
        }

        let shape: [i64; 2] = [
            (seq_len + self.seq_options.forward_skips).try_into()?,
            num_cols.try_into()?,
        ];
        let t = Tensor::from_slice(&data).view(shape);

        let return_seq_len = cur_offset - self.seq_options.forward_skips;
        let t_x = t.f_slice(0, Some(0), Some(return_seq_len.try_into()?), 1)?;
        let t_y = t.f_slice(
            0,
            Some(self.seq_options.forward_skips.try_into()?),
            Some((return_seq_len + self.seq_options.forward_skips).try_into()?),
            1,
        )?;

        self.last_rows
            .clone_from_slice(&data[(data.len() - num_cols * self.seq_options.forward_skips)..]);

        Ok(Some((t_x, t_y)))
    }
}

impl<TensorValType: TensorValTypeTrait, ParquetValType: ParquetValTypeTrait> ParquetToTorchSeqReader
    for ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
where
    ParquetValType::T: NumCast,
{
    fn get_tensors(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        self.get_tensor_len(self.seq_options.max_seq_len)
    }

    fn eof(&mut self) -> Result<bool> {
        self.advance_row_groups()?;
        Ok(self.iter.row_group_num >= self.num_row_groups)
    }
}

impl<TensorValType: TensorValTypeTrait, ParquetValType: ParquetValTypeTrait> Iterator
    for ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
where
    ParquetValType::T: NumCast,
{
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        match self.get_tensor_len(self.seq_options.max_seq_len) {
            Ok(tensors) => tensors,
            Err(e) => panic!("Error in iterator: {e}"),
        }
    }
}
