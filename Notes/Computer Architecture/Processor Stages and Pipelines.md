# In order processor pipelines

![[image-in_order_pipeline_of_cpu.png]]

## How a processor works

Processor can be divided into 5 logical stages:
1. Instruction Fetch
2. Operand Fetch
3. Execute
4. Memory Access
5. Register Write-back

## Pipeline and Hazards

Problem with this design: Only one stage of chip is active at any time out of 5.
Solution: Make each phase process a different instruction.

**Problem**: How to we make each step not interfere with others
Add pipeline latches. These are negative edge-triggered flip flops at the end of each step (shown in green). The store output of each step of pipeline so that it can be used in the input of the next step (in the next timeframe).

**Data Hazard**: Instruction A does a write on some register, before it's register write-back is done next Instruction B reaches operand-fetch and reads the old value.
Naive idea, add bubbles (no-ops) whenever such data dependency is noted. We may need to add atmost 3 no-ops, processor decides when to stall.

**Problem**: Data dependencies are pretty common, no-ops are expensive.
Our previous solution is not the best, so what we can do is feed information not just from the register file but also from the intermediate outputs, and we can multiplex these sources. We do this *forwarding as late as possible*, and values get forwarded to any prior-stage with future-instruction that needs it. Here we have 4 forwarding multiplexers: RW $\rightarrow$ MA, RW $\rightarrow$ EX, RW $\rightarrow$ OF, RW $\rightarrow$ EX.

**Load-Use Hazard**: Load produces output one cycle late (because it comes from memory access as opposed to others which come from execute). So it is one instruction cycle too late for the next instruction's execute step.
Nothing other than stalling can be done, which has to be done by a stall controller which can see the whole pipeline.

**Control Hazard**: A branch instruction is followed by other instructions I, if the branch is taken the other instructions should not have been executed, but they will get through some stages of the pipeline
We need to convert the incorrectly taken instructions to No-Ops retroactively. We do this by setting a bit in the instruction packet which does that.

## Measuring Performance

Instructions are of 2 types:
1. Static Instructions: Number of instructions that occur in code, code in loop counted only once.
2. Dynamic Instructions: Number of instructions that are actually executed, things in loop counted the number of times the loop runs.
We will use dynamic instructions to measure perfomance.

Performance is usually computed over several runs of several programs.
Performance is measured relative to other processors, usually taking a Geometric mean of rations of performance on each program.

Performance is:
$$P = \frac{1}{\text{\#secs}} = \frac{\text{\#ins}}{\text{\#cycle}} \cdot \frac{\text{\#cycles}}{\#secs}\cdot\frac{1}{\text{\#ins}} = \frac{\text{IPC} \times f}{\text{\#ins}}$$

How to get high frequency:
* Use smaller, power efficient transistors. Power dissipation is approximately proportional to cube of frequency. There are other ways to reduce power dissipation as well, discussed later.
* Have more pipeline stages, now that each stage is smaller, we can run the clock many times. IPC (instructions per cycle) is one for non-pipelined processor, and less than one for pipelined, but clock speed can increase a lot on pipelining.
  *However, increasing pipelines anymore leads to more stall, more latch delays, and therefore we don't get frequency gains. We haven't seen a large frequency increase since about 3GHz in 2005.*

How to get higher IPC:
* Depends on Architecture, things like value forwarding help reduce stall cycles.
* Depends on Compiler, it can rearrange code to loose less cycles to data and control hazards