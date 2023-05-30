
use std::sync::Arc;
use std::thread;
use crossbeam_channel::{bounded, select, Receiver, Sender};

enum ControlMessage {
    Stop,
}

pub struct BufferedIterator<ItemType>
 {
    buf_size: usize,
    item_queue: Receiver<Option<ItemType>>,
    control_queue: Sender<ControlMessage>,
    thread: thread::JoinHandle<()>,
}

struct BufferedIteratorWorker<IterType, ItemType> {
    item_queue: Sender<Option<ItemType>>,
    control_queue: Receiver<ControlMessage>,
    input_iter: IterType,
}

impl<InputType, IterType, ItemType> BufferedIteratorWorker<IterType, ItemType> 
//where OutputType: Send + 'static,
//   IterType: Iterator<Item = InputType> + Send,
{
    fn worker(&self) -> () {
        loop {
            let val = self.input_iter.next();
            let eof = val.is_none();

            select! {
                send(self.item_queue, val) -> res => res.expect("Failure sending"),
                recv(self.control_queue) -> _msg => return,
            }

            if eof { return }
        }

    }
}

impl<ItemType> BufferedIterator<ItemType>
//where OutputType: Send + 'static 
{
    pub fn new<IterType, InputType: Send>(
        mut input_iter: IterType,
        buf_size: usize
    ) -> Self
    //where FnType: Fn(InputType) -> OutputType + Send + Sync,
    //    IterType: Iterator<Item = InputType> + Send 
    {

        let (
            item_queue_sender,
            item_queue_receiver
        ) = bounded::<Option<OutputType>>(buf_size);
        let (
            control_queue_sender,
            control_queue_receiver
        ) = bounded::<ControlMessage>(1);

        let worker = BufferedMapperWorker {
            item_queue: item_queue_sender,
            control_queue: control_queue_receiver,
            input_iter,
        };

        let thread = thread::spawn(move || {
            worker.worker()
        });

        BufferedIterator {
            buf_size,
            item_queue: item_queue_receiver,
            control_queue: control_queue_sender,
            thread,
        }
    }
}

impl<ItemType> Drop for BufferedIterator<ItemType> {
    fn drop(&mut self) {
        self.control_queue.send(ControlMessage::Stop);
        self.thread.join();
    }
}

impl<ItemType> Iterator for BufferedIterator<ItemType> 
where ItemType: 'static {
    type Item = ItemType;

    fn next(&mut self) -> Option<ItemType> {
        self.item_queue.recv().expect("Failure to receive item")
    }
}