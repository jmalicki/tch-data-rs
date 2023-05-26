
use std::path::Path;
use tch::Tensor;
use threadpool::ThreadPool;
use std::sync::Arc;

use crossbeam_channel::{bounded, Sender, Receiver, select};

use crate::parquet_tensors::{ParquetToTorchSeqReader, ParquetToTorchSeqReaderFactory, ParquetLSTMTensorError, Result};

enum ControlMessage {
    Stop,
}

enum WorkQueueMessage {
    NewFile(Path),
}

enum RoundRobinError {
    ParquetLSTMTensorError(ParquetLSTMTensorError),
    StopWorker,
}

type Result<T> = core::result::Result<T, BatcherError>;

struct RoundRobinWorker<Item>
where Item: Send {
    item_queue: Sender<Option<Item>>,
    work_queue: Receiver<WorkQueueMessage>,
    control_queue: Receiver<ControlMessage>,
    reader_create: Arc<ParquetToTorchSeqReaderFactory>,
}

impl<Item> RobinWorker<Item> 
  where Item: Send
{
    // Returns Ok(()) on eof, otherwise error
    fn get_tensor_loop(&mut self, reader: &dyn ParquetToTorchSeqReader) -> Result<()> {
        loop {
            // the select! macro returns a random item if multiple are ready,
            // so make sure we always respect a stop message
            match self.control_queue.try_recv() {
                Ok(ControlMessage::Stop) => return  Err(BatcherError::Stop),
                _ => (),
            }

            let item = reader.get_tensors();
            let eof = tensors.is_none();

            select! {
                send(self.item_queue, item) -> res => res?,
                recv(self.control_queue) -> msg => return Err(BatcherError::Stop),
            }

            if eof { return }
        }
    }

    fn worker_loop(&mut self) -> Result<()> {
        loop {
            let path = select! {
                recv(self.work_queue) -> msg => match {
                    NewFile(path) => path,
                    _ => return Ok(()),
                },
                recv(self.control_queue) -> msg => return Ok(()),
            };

            let reader = self.reader_create(&path)?;
            match self.get_tensor_loop(&reader) {
                Err(BatcherError::Stop) => return Ok(()),
                Err(e) => return Err(e),
                Ok(()) => (),
            };
            drop(reader);
        }
    }

    pub fn worker(&mut self) -> Result<()> {
        let res = self.worker_loop();

        self.work_queue.close();
        self.control_queue.close();
        self.tensor_queue.close();

        res
    }
}

struct TensorRoundRobin {
    round_robin_size: usize,
    buffer_size: usize,
    filenames: Vec<Path>,
    reader_factory: Arc<ParquetToTorchSeqReaderFactory>,
}

struct RobinIterator<Item>  {
    parent: &RoundRobin<Item>,
    next_file: usize,
    next_queue: usize,
    tensor_queues: Vec<Receiver<Option<Item>>>,
    threadpool: ThreadPool,
    work_queue: Sender<WorkQueueMessage>,
    control_queue: Sender<ControlMessage>,
}

impl<Item> Iterator for TensorRoundRobinIterator<Item> {
    type Item = Item;

    fn next(&mut self) -> Option<Item> {
        loop {
            // check for end of epoch
            if item_queues.empty() { return None }

            match self.item_queues[self.next_queue].recv() {
                Err(e) => panic!("Error dequeuing tensor: {e}"),
                Ok(Some(item)) => {
                    self.next_queue += 1;
                    if self.next_queue == self.item_queues.len() { self.next_queue = 0 }
                    return Ok(item)
                },
                Ok(None) => (),
            }

            // EOF for that file, send next file if we have more

            if self.next_file < self.parent.filenames.len() {
                self.work_queue.send(
                    WorkQueueMessage::NewFile(self.parent.filenames[self.next_file].copy())
                ).expect("Error sneding new work queue item");
                self.next_file += 1;
                self.next_queue += 1;
            } else {
                tensor_queues[self.next_queue].close();
                tensor_queues.remove(self.next_queue);
            }
            if self.next_queue == tensor_queues.len() { self.next_queue = 0 }
        }
    }
}

impl<Item> Drop for RoundRobinIterator<Item> {

    fn drop(&mut self) {
        for i in 0..self.params.round_robin_size {
            self.control_queue.send(ControlMessage::Stop)
        }
        self.threadpool.join();
        for queue in self.item_queues {
            queue.close()
        }
        self.work_queue.close();
        self.control_queue.close();
    }
}

impl<Item> RoundRobin<Item> {
    pub fn new<StrRef: AsRef<str>>(
        filenames: &[StrRef],
        round_robin_size: usize,
        buffer_size: usize,
        reader_create: ParquetToTorchSeqReaderFactory,
    ) -> Result<RoundRobin> {
        let res = RoundRobin {
            filenames: filenames.map(|s| s.to_string()).collect(),
            round_robin_size,
            buffer_size,
            reader_create,
        };

        Ok(res)
    }


}