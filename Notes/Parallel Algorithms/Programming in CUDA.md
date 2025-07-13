A CUDA kernel spawns on many threads and blocks.

* Threads: Individual units of computation where processing is done sequentially, has private access to it's own register variables and not to those of any other threads.
* Blocks: Groups of threads that have access to the same L1 cache, and are colocated on the same streaming-multiprocessor. They share memory and other resources to work together. Each block can have upto 1024 threads (due to a limit on the number of cores in a streaming-multiprocessor).

![[image-nvidia_memory_heirachy_with_sm.png]]