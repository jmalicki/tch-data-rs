use std::path::PathBuf;
use std::sync::Arc;
use threadpool::ThreadPool;

use crossbeam_channel::{bounded, select, Receiver, Sender};
use thiserror::Error;

use crate::parquet_tensors::{ParquetLSTMTensorError, ParquetToTorchSeqReaderFactory};

enum ControlMessage {
    Stop,
}

enum WorkQueueMessage {
    NewFile(PathBuf),
}

#[derive(Debug, Error)]
pub enum RoundRobinError {
    #[error(transparent)]
    ParquetLSTMTensorError(#[from] ParquetLSTMTensorError),
    #[error("Worker stopped")]
    StopWorker,
    #[error("Crossbeam error: {error}")]
    CrossbeamError { error: String },
}

impl<Item: Send> From<crossbeam_channel::SendError<Item>> for RoundRobinError {
    fn from(e: crossbeam_channel::SendError<Item>) -> RoundRobinError {
        RoundRobinError::CrossbeamError {
            error: e.to_string(),
        }
    }
}

type Result<T> = core::result::Result<T, RoundRobinError>;

struct RoundRobinWorker<Item>
where
    Item: Send,
{
    item_queue: Sender<Option<Item>>,
    work_queue: Receiver<WorkQueueMessage>,
    control_queue: Receiver<ControlMessage>,
    reader_create: Arc<ParquetToTorchSeqReaderFactory<Item>>,
}

impl<'a, Item> RoundRobinWorker<Item>
where
    Item: Send,
    RoundRobinError: From<crossbeam_channel::SendError<Option<Item>>>,
{
    // Returns Ok(()) on eof, otherwise error
    fn get_tensor_loop(&mut self, reader: &mut dyn Iterator<Item = Item>) -> Result<()> {
        loop {
            // the select! macro returns a random item if multiple are ready,
            // so make sure we always respect a stop message
            match self.control_queue.try_recv() {
                Ok(ControlMessage::Stop) => return Err(RoundRobinError::StopWorker),
                _ => (),
            }

            let item = reader.next();
            let eof = item.is_none();

            select! {
                send(self.item_queue, item) -> res => res?,
                recv(self.control_queue) -> _msg => return Err(RoundRobinError::StopWorker),
            }

            if eof {
                return Ok(());
            }
        }
    }

    fn worker_loop(&mut self) -> Result<()> {
        loop {
            let path = select! {
                recv(self.work_queue) -> msg => match msg {
                    Ok(WorkQueueMessage::NewFile(path)) => path,
                    _ => return Ok(()),
                },
                recv(self.control_queue) -> _msg => return Ok(()),
            };

            let mut reader = (self.reader_create)(&path)?;
            match self.get_tensor_loop(&mut *reader) {
                Err(RoundRobinError::StopWorker) => return Ok(()),
                Err(e) => return Err(e),
                Ok(()) => (),
            };
            drop(reader);
        }
    }

    pub fn worker(&mut self) -> Result<()> {
        self.worker_loop()
    }
}

pub struct RoundRobin<Item> {
    round_robin_size: usize,
    buffer_size: usize,
    filenames: Vec<PathBuf>,
    reader_create: Arc<ParquetToTorchSeqReaderFactory<Item>>,
}

// derive(Clone) did not work
impl<Item> Clone for RoundRobin<Item> {
    fn clone(&self) -> Self {
        RoundRobin {
            round_robin_size: self.round_robin_size,
            buffer_size: self.buffer_size,
            filenames: self.filenames.clone(),
            reader_create: self.reader_create.clone(),
        }
    }
}

pub struct RoundRobinIterator<Item> {
    round_robin_size: usize,
    next_file: usize,
    next_queue: usize,
    item_queues: Vec<Receiver<Option<Item>>>,
    threadpool: ThreadPool,
    work_queue: Sender<WorkQueueMessage>,
    control_queue: Sender<ControlMessage>,
    parent: RoundRobin<Item>,
}

impl<Item: Send + 'static> RoundRobinIterator<Item> {
    pub fn new(parent: &RoundRobin<Item>) -> Result<RoundRobinIterator<Item>> {
        let threadpool = ThreadPool::new(parent.round_robin_size);
        let mut item_queues = Vec::<Receiver<Option<Item>>>::with_capacity(parent.round_robin_size);
        let (work_queue_sender, work_queue_receiver) = bounded(parent.round_robin_size);
        let (control_queue_sender, control_queue_receiver) = bounded(parent.round_robin_size);

        let mut next_file = 0;
        while next_file < parent.round_robin_size && next_file < parent.filenames.len() {
            let (sender, receiver) = bounded::<Option<Item>>(parent.buffer_size);
            item_queues.push(receiver);

            let mut worker = RoundRobinWorker {
                item_queue: sender,
                work_queue: work_queue_receiver.clone(),
                control_queue: control_queue_receiver.clone(),
                reader_create: parent.reader_create.clone(),
            };

            work_queue_sender.send(WorkQueueMessage::NewFile(
                parent.filenames[next_file].clone(),
            ))?;

            threadpool.execute(move || {
                worker.worker().expect("RoundRobinWorker error");
            });

            next_file += 1;
        }

        Ok(RoundRobinIterator {
            round_robin_size: parent.round_robin_size,
            next_file,
            next_queue: 0,
            item_queues,
            threadpool,
            work_queue: work_queue_sender,
            control_queue: control_queue_sender,
            parent: parent.clone(),
        })
    }
}

impl<Item> Iterator for RoundRobinIterator<Item> {
    type Item = Item;

    fn next(&mut self) -> Option<Item> {
        loop {
            // check for end of epoch
            if self.item_queues.is_empty() {
                return None;
            }

            match self.item_queues[self.next_queue].recv() {
                Err(e) => panic!("Error dequeuing tensor: {e}"),
                Ok(Some(item)) => {
                    self.next_queue += 1;
                    if self.next_queue == self.item_queues.len() {
                        self.next_queue = 0
                    }
                    return Some(item);
                }
                Ok(None) => (),
            }

            // EOF for that file, send next file if we have more

            if self.next_file < self.parent.filenames.len() {
                self.work_queue
                    .send(WorkQueueMessage::NewFile(
                        self.parent.filenames[self.next_file].clone(),
                    ))
                    .expect("Error sending new work queue item");
                self.next_file += 1;
                self.next_queue += 1;
            } else {
                self.item_queues.remove(self.next_queue);
            }
            if self.next_queue == self.item_queues.len() {
                self.next_queue = 0
            }
        }
    }

    /// leaving this here because we need to make upstreams iterators etc.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<Item> Drop for RoundRobinIterator<Item> {
    fn drop(&mut self) {
        for _i in 0..self.round_robin_size {
            self.control_queue.send(ControlMessage::Stop).expect("Error sending stop message");
        }
        self.threadpool.join();
    }
}

impl<Item> RoundRobin<Item> {
    pub fn new<StrRef: AsRef<str> + std::convert::AsRef<std::ffi::OsStr>>(
        filenames: &[StrRef],
        round_robin_size: usize,
        buffer_size: usize,
        reader_create: Arc<ParquetToTorchSeqReaderFactory<Item>>,
    ) -> Result<RoundRobin<Item>> {
        let res = RoundRobin {
            filenames: filenames.iter().map(|s| PathBuf::from(s)).collect(),
            round_robin_size,
            buffer_size,
            reader_create: reader_create,
        };

        Ok(res)
    }
}

impl<Item: Send + 'static> IntoIterator for &RoundRobin<Item> {
    type Item = Item;
    type IntoIter = RoundRobinIterator<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self).expect("Iterator creation error")
    }
}
