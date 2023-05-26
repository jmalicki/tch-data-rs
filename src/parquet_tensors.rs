
use parquet::data_type::DataType;
use parquet::{
    column::reader::ColumnReaderImpl,
    file::{
        metadata::ParquetMetaData,
        reader::{FileReader, RowGroupReader, SerializedFileReader},
    },
};
use num::NumCast;
use std::{fs::File, path::Path, usize};
use std::collections::HashMap;
use tch::{TchError, Tensor};
use thiserror::Error;
use core::hash::Hash;
use std::cmp::Eq;

// Keep track of iterator over parquet file
struct ParquetToTorchSeqReaderIterator<ParquetValType>
    where ParquetValType: DataType,
{
    col_readers: Vec<Option<ColumnReaderImpl<ParquetValType>>>,
    row_group_num: usize,
    row_group_offset: usize,
    row_group_num_rows: usize,

}

pub type ParquetToTorchSeqReaderFactory = dyn Fn(&Path) -> Box<dyn ParquetToTorchSeqReader>;

pub trait ParquetToTorchSeqReader {
    fn get_tensors(&mut self) -> Result<Option<(Tensor, Tensor)>>;
    fn eof(&mut self) -> Result<bool>;
}

pub struct ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
    where
      ParquetValType: DataType
{
    reader: SerializedFileReader<File>,
    col_indices: Vec<usize>,
    max_levels: Vec<i16>,
    num_row_groups: usize,
    max_seq_len: usize,
    iter: ParquetToTorchSeqReaderIterator<ParquetValType>,
    // number of steps to skip between input and prediction
    forward_skips: usize,
    augment_offset: bool,
    last_rows: Vec<TensorValType>,
}

impl<ParquetValType: DataType> ParquetToTorchSeqReaderIterator<ParquetValType> {
    fn new(row_group_num: usize, num_cols: usize) -> Self {
        ParquetToTorchSeqReaderIterator {
            // row_group: None,
            row_group_num,
            row_group_num_rows: 0,
            row_group_offset: 0,
            col_readers: {
                let mut  v = Vec::new();
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
    IoError(#[from] std::io::Error)
}

type Result<T> = core::result::Result<T, ParquetLSTMTensorError>;

impl<TensorValType, ParquetValType: DataType> ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType> where
    TensorValType: tch::kind::Element + Default + NumCast,
    ParquetValType::T: Copy + Default + NumCast {
    /// Return indices for given column names in parquet file
    ///
    /// # Arguments
    ///
    /// * `metadata` - Metadata for this parquet file
    /// * `colnames` - Array of string slices in the order that indices should be returned
    ///
    /// # Returns
    /// List of indices of columns in same order as `colnames`
    fn column_indices_and_levels<StrRef: AsRef<str> + Hash + Eq>(
        metadata: &ParquetMetaData,
        colnames: &[StrRef]
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
        max_seq_len: usize,
        forward_skips: usize,
        augment_offset: bool,
    ) -> Result<Self> {
        let f = File::open(filename)?;
        let num_cols = colnames.len();

        let reader = SerializedFileReader::new(f)?;
        let parquet_metadata = reader.metadata();

        let (col_indices, max_levels) = Self::column_indices_and_levels(parquet_metadata, colnames)?;
        let num_row_groups = parquet_metadata.num_row_groups();

        let iter_info = Self::next_row_group(&reader, 0, num_row_groups, &col_indices)?;

        let mut res = ParquetToTorchSeqReaderImpl {
            reader,
            num_row_groups,
            col_indices,
            max_levels,
            iter: iter_info,
            max_seq_len,
            forward_skips,
            augment_offset,
            last_rows: Vec::new(),
        };

        let mut last_rows = vec![TensorValType::default(); forward_skips * num_cols];
    
        // If we couldn't read enough, we will be at eof
        let num_read = res.read_data(&mut last_rows, 0, forward_skips)?;
        // Have to move the vector so we don't double borrow
        res.last_rows = last_rows;
    
        if num_read < forward_skips && !res.eof()? {
            panic!("Only read {num_read} values out of {forward_skips}, but eof not triggered, in init!")
        }

        Ok(res)
    }

    fn init_col_readers(
        row_group_reader: &dyn RowGroupReader,
        col_indices: &[usize],
    ) -> Result<Vec<
            Option<ColumnReaderImpl<ParquetValType>>
    >> {
        let readers = col_indices.iter().copied()
            .map(|idx|
                row_group_reader.get_column_reader(idx).map(
                    |col_reader| Some(parquet::column::reader::get_typed_column_reader::<ParquetValType>(col_reader))
                )
            ).collect::<core::result::Result<Vec<_>, _>>()?;
        Ok(readers)
    }

    fn next_row_group(
        reader: &dyn FileReader,
        row_group_num: usize,
        num_row_groups: usize,
        col_indices: &[usize],
    ) -> Result<ParquetToTorchSeqReaderIterator<ParquetValType>> {
        if row_group_num >= num_row_groups {
            Ok(ParquetToTorchSeqReaderIterator::new(row_group_num, col_indices.len()))

        } else {
            let row_group = reader.get_row_group(row_group_num)?;
            let row_group_num_rows = row_group.metadata().num_rows().try_into()?;
            let col_readers = Self::init_col_readers(&*row_group, col_indices)?;

            Ok(ParquetToTorchSeqReaderIterator {
                // row_group: Some(row_group),
                row_group_num,
                row_group_num_rows,
                row_group_offset: 0,
                col_readers
            })
        }
    }

    fn advance_row_groups(&mut self) -> Result<()> {
        // check row group
        while self.iter.row_group_offset == self.iter.row_group_num_rows {
            panic!("Untested advance_row_groups");
            if self.iter.row_group_num == self.num_row_groups { 
                return Ok(())
            }

            self.iter = Self::next_row_group(
                &self.reader,
                self.iter.row_group_num + 1,
                self.num_row_groups,
                &self.col_indices,
            )?;
        }

        Ok(())
    }

    fn read_row_group_data (
        &mut self,
        data: &mut [TensorValType],
        cur_offset: usize,
        seq_len: usize,
    ) -> Result<usize> {
        let num_cols: usize = self.col_indices.len();

        if seq_len * num_cols > data.len() {
            panic!(
                "Data requested {data_req} larger than buffer {data_len}",
                data_req=seq_len * num_cols, data_len=data.len(),
            )
        }

        self.advance_row_groups()?;
        if cur_offset > seq_len { panic!("offset {cur_offset} greater than sequence length {seq_len}") }

        let num_rows: usize = core::cmp::min(
            seq_len - cur_offset,
            self.iter.row_group_num_rows - self.iter.row_group_offset,
        );

        if num_rows == 0 { return Ok(0) } // EOF

        let mut buffer: Vec<ParquetValType::T> = vec![ParquetValType::T::default(); num_rows];
        let mut def_levels: Vec<i16> = vec![0; num_rows];
        let mut rep_levels: Vec<i16> = vec![0; num_rows];

        for (col_idx, col_reader) in self.iter.col_readers.iter_mut().enumerate() {
            if let Some(reader) = col_reader.as_mut() {
                let (non_null_read, levels_read) =
                    reader.read_batch(
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

                // Now we copy non-null bytes
                let mut src_idx: usize = 0;
                let max_level = self.max_levels[col_idx];


                // we have a level for every position, but a buffer in src_idx only for non-nulls
                for dest_idx in 0..levels_read {
                    if def_levels[src_idx] == max_level {
                        if src_idx < non_null_read {  // defensive if for later panic
                            data[(cur_offset + dest_idx) * num_cols + col_idx] =
                                num::cast(buffer[src_idx]).unwrap()
                        }
                        src_idx += 1;

                        if src_idx > non_null_read {
                            panic!(
                                "More non-null values ({src_idx}) than reported read {non_null_read}!"
                            )    
                        }
                    }
                }
            }
        }

        self.iter.row_group_offset += num_rows;

        Ok(num_rows)
    }

    pub fn read_data(&mut self, data: &mut [TensorValType], cur_offset: usize, seq_len: usize) -> Result<usize> {
        let mut rows_read = 0;

        while cur_offset + rows_read < seq_len {
            let num_rows = self.read_row_group_data(data, cur_offset + rows_read, seq_len)?;
            rows_read += num_rows;
            if num_rows == 0 { break; }
        }

        Ok(rows_read)
    }

    pub fn get_tensor_len(&mut self, seq_len: usize) -> Result<Option<(Tensor, Tensor)>> {
        let num_cols: usize = self.col_indices.len();
        let mut data: Vec<TensorValType> = vec![TensorValType::default(); (self.forward_skips + seq_len) * num_cols];

        data[..(num_cols*self.forward_skips)].clone_from_slice(&self.last_rows);

        let mut cur_offset: usize = self.forward_skips;
    
        let rows_read = self.read_data(&mut data, cur_offset, seq_len + self.forward_skips)?;
        cur_offset += rows_read;

        if cur_offset == self.forward_skips {
            return Ok(None)
        }

        let shape: [i64; 2] = [(seq_len + self.forward_skips).try_into()?, num_cols.try_into()?];
        let t = Tensor::from_slice(&data).view(shape);

        let return_seq_len = cur_offset - self.forward_skips;
        let t_x = t.f_slice(
            0,
            Some(0),
            Some(return_seq_len.try_into()?),
            1,
        )?;
        let t_y = t.f_slice(
            0,
            Some(self.forward_skips.try_into()?),
            Some((return_seq_len+self.forward_skips).try_into()?),
            1,
        )?;

        self.last_rows.clone_from_slice(
            &data[(data.len() - num_cols * self.forward_skips)..]);

        Ok(Some((t_x, t_y)))
    }


}

impl<TensorValType, ParquetValType> ParquetToTorchSeqReader for ParquetToTorchSeqReaderImpl<TensorValType, ParquetValType>
  where ParquetValType: DataType,
    TensorValType: tch::kind::Element + Default + NumCast,
    ParquetValType::T: Copy + Default + NumCast {
    fn get_tensors(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        self.get_tensor_len(self.max_seq_len)
    }

    fn eof(&mut self) -> Result<bool> {
        self.advance_row_groups()?;
        return Ok(self.iter.row_group_num == self.num_row_groups);
    }
}

#[cfg(test)]
mod test {
    use std::{fs, path::Path, sync::Arc};

    use parquet::{
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::parser::parse_message_type,
    };

    static message_type: String = format!(
        "
    message schema {
      REQUIRED INT32 b;
    }
  "
    );

    fn create_file(path: &Path, message_type: &str, num_row_groups: usize) {
        let schema = Arc::new(parse_message_type(message_type).unwrap());
        let props = Arc::new(WriterProperties::builder().build());
        let file = fs::File::create(&path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
        let mut row_group_writer = writer.next_row_group().unwrap();

        let mut i: i32 = 0;

        while let Some(mut col_writer) = row_group_writer.next_column().unwrap() {
            // ... write values to a column writer
            col_writer.untyped().col_writer.close().unwrap()
        }
        row_group_writer.close().unwrap();
        writer.close().unwrap();
    }

    static path: Path = Path::new("/path/to/sample.parquet");

    //let bytes = fs::read(&path).unwrap();
    //assert_eq!(&bytes[0..4], &[b'P', b'A', b'R', b'1']);
}
