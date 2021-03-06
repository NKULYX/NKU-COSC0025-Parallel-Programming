# 问题描述

在进行科学计算的过程中，经常会遇到对于多元线性方程组的求解，而求解线性方程组的一种常用方法就是Gauss消去法。即通过一种自上而下，逐行消去的方法，将线性方程组的系数矩阵消去为主对角线元素均为1的上三角矩阵。
$$
\left[
\begin{matrix}
a_{11}&a_{12}&\cdots&a_{1\ n-1}&a_{1n}\\
a_{21}&a_{22}&\cdots&a_{2\ n-1}&a_{2n}\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
a_{n-1\ 1}&a_{n-1\ 2}&\cdots&a_{n-1 \ n-1}&a_{n-1\ n}\\
a_{n\ 1}&a_{n\ 2}&\cdots&a_{n \ n-1}&a_{n\ n}\\
\end{matrix}
\right]
=>
\left[
\begin{matrix}
1&a_{12}'&\cdots&a_{1\ n-1}'&a_{1n}'\\
0&1&\cdots&a_{2\ n-1}'&a_{2n}'\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&\cdots&1&a_{n-1\ n}\\
0&0&\cdots&0&1\\
\end{matrix}
\right]
$$
在整个消去的过程中，主要包含两个过程

![image-20220428214131431](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220428214131431.png)

# 实验设计

考虑Gauss消去的整个过程中主要涉及到两个阶段，一个是在消元行行内除法过程，一个是其余行减去消元行的过程。而就每个阶段而言，其所做的工作基本是一致的，只是在不同的消元轮次时，消元的起始位置不同。尤其是针对第二个阶段，即其余行依次减去消元行的过程，这个阶段每一行所做的工作是完全一致的，十分适合并行化处理，即将待消去的行平均分配给几个线程，由于这些数据之间不存在依赖性，因此每个线程只需要各自完成好自己的工作即可，不存在线程之间进行通信的额外开销。

而对于第一阶段，即消元行行内进行除法操作时，由于这个问题规模相对较小，如果将待操作的数据分配给不同的线程进行处理的话，线程挂起到唤醒这部分的时间开销相较于要处理的问题而言占比很高，因此不适合进行多线程并行处理，但是仍可以结合SIMD的向量化处理。同样在第二阶段，被消元行依次减去消元行的过程中，每一行内的减法运算同样也不适合进行多线程的并行处理，也可以采用SIMD进行向量化处理。

在本次实验中，将设计以下实验进行探究：

## pthread并行处理

对于Gauss消去的过程，在每一轮消去中主要包含两个阶段，首先是针对消元行做除法运算，然后是对于剩余的被消元行，依次减去消元行的某个倍数。在每一轮的过程中，除法操作和消元减法操作之间是有着严格的先后顺序的，即必须首先完成消元行的除法操作之后，才能够执行被消元行的减法操作。因此需要引入信号量进行同步控制，即当0号线程完成了对于消元行的除法操作之后，依次向其余挂起等待的线程发送信号，之后所有线程一起并行执行被消元行的减法操作。

在执行消去的时候，考虑对于数据采用分散的划分方式，即以线程数量为步长对于剩余的被消元行进行划分，分配给不同的线程。不同的线程之间执行的工作是完全一致的，并且由于不同行之间并不存在数据依赖，因此可以避免线程之间的通信开销。

而由于不同的线程在执行消去操作时所需要的时间可能并不相同，因此需要在所有线程完成本轮被分配的消元任务之后，进行一次同步控制。在这一次同步控制中，出于方便考虑，使用了barrier进行同步控制。即只有当所有的线程都完成了消去任务之后，才会进入下一轮的消元。

## 数据块划分设计

在进行任务划分时，给出的样例中采用了等步长的划分方式，这种划分方式存在一定的弊端。即当数据规模比较大的时候，由于L1 cache大小有限，很有可能会导致在访问下一个间隔为线程数的行的时候出现cache miss，这样就需要到L2、L3甚至内存中去读取数据。这将会造成额外的访存开销。

因此在进行数据划分的时候，考虑设计一种充分利用cache优化的数据划分方式，即将数据按块划分。每个线程负责连续的几行的消去任务。这样做的好处是，当线程正在处理当前行的时候，CPU可能会提前预取下一行的数据到cache中，这就会使得下一次进行数据访问的时候，能够尽快在cache中命中，减少了不必要的访存开销。

## 数据动态划分设计

考虑在进行任务划分的时候，由于不同线程在执行任务的时候，所需要的时间可能不一致，甚至因为数据规模不是线程数量的整数倍，导致某些线程出现在个别轮次中处于空等待的状态。这是由于数据划分的时候，由于细粒度的数据划分导致的线程之间负载不均衡。

因此考虑在设计数据划分的时候采用动态的数据划分方式。即在对被消元行执行减法操作的过程中，并不明确指定某个线程对哪部分数据执行任务，而是根据各个线程任务完成的情况动态的进行数据划分。即通过一个全局的变量index来指示现阶段已经处理到哪一行。而当某一个线程完成了其被分配的任务的时候，会查看关于index的互斥量，如果这个互斥量并没有上锁，则说明当前处于可以进行任务划分的阶段。于是让这个线程对关于index的互斥量上锁，并将index所指的行分配给该线程，任务分配完成后，线程释放互斥量，然后去执行所分配的任务。

这样就可以保证每条线程都一直在执行被分配的任务，而不会出现个别线程由于负载不均衡出现空等待的现象，而其他线程还在执行任务。由于只有当所有线程的任务都执行完毕的时候才会进入下一轮迭代，因此那些进行空等待的线程就浪费了CPU的计算资源。这就是该实验设计选择进行优化的方向。

## 不同数据规模和线程数下的性能探究

考虑到线程的创建，调度，挂起和唤醒等操作相对于简单的计算操作而言，所需要的时间开销是非常大的。因此可以推测，当问题规模比较小的时候，由于线程调度导致的额外开销会抵消掉多线程优化效果，甚至还会表现出多线程比串行算法更慢的情况。而随着问题规模的增加，线程之间调度切换所需要的时间开销相对于线程完成任务所需要的时间而言已经占比很低，这样就能够正常反映出多线程并行优化的效果。因此，设计实验探究在不同数据规模下，多线程并行优化算法的优化效果。此外还将探究在所使用的线程数量不同的情况下，并行算法优化效果的变化情况。

## x86平台迁移

本次实验除了对ARM架构下采用neon指令集架构结合pthread多线程编程，对Gauss消去算法进行并行化处理，还将算法迁移到了x86平台上，采用x86中的SSE、AVX和AVX512指令集架构分别对算法进行重构，然后对比实验效果。

# 实验结果分析

## ARM平台

### pthread并行处理

为了能够探究pthread并行算法的优化效果，考虑调整问题规模，测量在不同任务规模下，pthread并行优化算法对于普通串行算法和SIMD向量化优化算法的加速比。在本次实验中，pthread并行算法中，同样融合了SIMD的向量化处理。在ARM平台上，SIMD的实现是基于Neon指令集架构的。为了能够比较全面的展现并行优化效果随问题规模的变化情况，在问题规模小于1000时采用步长为100，而当问题规模大于1000时，步长调整为1000。三种算法的在不同问题规模下的表现如下表所示。

在实验设计时，SIMD进行向量化处理的时候，采用的是四路向量化处理，而pthread多线程优化时，总共开启了8条线程，其中一条线程负责除法操作，剩余的7条线程负责做消元操作。因此从时间表现情况来看，理论上SIMD优化算法所需要的时间应该是串行算法的$\frac{1}{4}$，pthread多线程所需要的时间应该是SIMD向量化的$\frac{1}{7}$。而从实验数据来看，当问题规模较小的时候，pthrad多线程算法的时间性能甚至差于普通的串行算法。这是由于线程的创建，挂起，唤醒和切换等操作，所需要消耗的时钟周期数要远远多余简单的运算操作。因此当问题规模较小时，由于运算操作在整个问题求解的过程中所占比例较低，因此线程额外开销的副作用就会显现出来。而随着问题规模的增大，pthread多线程的优势就能够显现出来。两种并行优化算法的加速比变化如下图所示。

从图像中可以看出，SIMD的加速比随着问题规模的增加基本保持稳定，由于算法中还涉及到其他的数据处理，因此其加速比只达到了2左右，并没有能够达到理论上的4。而pthread优化的效果则随着问题规模的增加呈现出持续上升的趋势。这是因为，问题规模的增加，使得程序在运行的过程中，运算所占比例不断上升，这将会逐步抵消由于线程切换导致的额外开销。从数据中可以看出，当问题规模达到2000时，已经接近了其对SIMD的理论加速比。可以推测，当问题规模持续上升时，这个加速比将会接近7。

### 数据划分方式对比

本次实验中，除了进行基础的pthread多线程优化尝试之外，还从数据划分的角度出发，考虑不同的数据划分方式，对于并行算法优化效果的影响。结合前文实验设计，分别对比了循环划分，块划分和动态划分三种方式，在不同问题规模下的表现效果，并以SIMD算法为baseline，其加速比变化情况如下图所示。

可以看到，随着问题规模的增大，这三种任务划分方式的加速比都是逐渐去接近理论加速比的。但是也可以注意到，在三种任务划分之间的性能表现还是存在着明显差异的。

从cache优化的角度出发，循环划分和块划分的主要区别就在于能都利用到cache优化。就循环划分这种方式而言，线程在处理完当前行之后，接下来要处理的行距离当前间隔为NUM_PHTREAD，因此当数据规模很大的时候，会因为L1 cache不能够容纳下足够的数据，或者由于CPU未能够及时的预取下一行数据，而导致cache miss，因此需要额外的访存开销。而块划分的方式就能够很好的弥补这一点，其原因是对于每个线程而言，他所需要处理的数据之间在内存上是连续的，因此有很好的cache优势，因此能够减小由于cache miss导致的额外访存开销。使用perf工具对于这两种算法的L1 cache的命中率进行检测，如下表所示。块划分的命中率能够达到98%，而循环划分的方式只有94%左右，两者差异不大，因此在性能表现上的差异也不显著。

从负载均衡的角度出发，循环划分和动态数据划分的主要区别就在于能否充分利用各个线程的计算资源，尽可能减少同步等待所导致的额外开销。如果采用循环数据划分的方式，由于各个线程完成任务所需要的时间不尽相同，并且由于问题规模可能不是线程的整数倍，因此可能存在某些线程较早完成任务进入同步等待状态，而其他线程还未完成任务，因此就浪费了一些计算资源。而动态数据划分就是从这个角度出发，尽可能充分利用每个线程的计算资源，使得任务能够在线程之间得到比较均匀的划分。从图中也可以看出，当问题规模较小的时候，动态划分方式的表现不如循环划分，这是由于动态划分在保证负载均衡的前提下，牺牲了线程调度的开销，由于每个线程不清楚自己具体的工作，因此会存在比较大的线程同步和线程切换的开销，这种额外开销在问题规模比较小的时候会格外显著。而当问题规模提升的时候，可以发现，动态划分方式的表现已经能够超越循环划分，负载均衡带来的收益已经抵消了线程调度的额外开销。

### 线程数量对比

本次实验中，还探究了pthread多线程优化方法，在开启不同的线程数量时，优化效果的变化情况。为了能够显著体现pthread的优化效果，选取数据规模为1000，调整线程数量，观测加速比的变化情况如下图所示。

从图像中可以看出，随着线程数量的线性增加，pthread多线程的优化效果也是呈现出线性提升的趋势。而当线程数量超过8个之后，其优化效果不再有显著的变化。这是由于实验使用的服务器单CPU核心能够提供8个线程，因此当线程数量小于8个的时候，CPU核心能够使用自己的8个线程调度任务，而当所需要的线程数量超过8个之后，就需要和服务器中的其他CPU核心借用线程，这之间会存在着额外的调度开销，因此抵消掉了性能的提升效果。

## x86平台迁移

### 多种SIMD指令集融合

基于前文在ARM平台上对于pthread多线程编程的探究，在本次实验中还将pthread多线程优化方法迁移到x86平台上，做同样的实验探究。由于x86平台上拥有更多的SIMD指令集架构，因此实验中分别探究了SSE、AVX和AVX512三种指令集架构配合pthread多线程的优化效果，测量在不同问题规模下的运行时间，如下表所示。可以看出，pthread多线程可以结合多种SIMD指令集架构，并且在各种指令集架构上的表现基本保持稳定，并没有出现在某种指令集架构下不能够发挥很好的多线程优势的现象。

此外，实验还以SSE指令集架构为例，探究了随着问题规模的变化，不同SSE向量化处理和pthread多线程结合SSE向量化处理这两种方法的表现情况，变化趋势图如下图所示。

可以看出，在问题规模小于1000的时候，加速比随着问题规模的线性增长呈现出一个线性上升的趋势。而当问题规模超过1000的时候，会发现加速比出现了一个下降的趋势。结合VTune性能分析工具分析的结果，分析其原因是因为，当问题规模增加时，超过了线程cache的大小，导致出现了大量的cache miss，额外的访存开销在一定程度上抵消了多线程的优化效果，使得加速比的变化出现拐点。

### 不同任务划分方式对比

从cache优化的角度出发，循环划分和块划分的主要区别就在于能都利用到cache优化。就循环划分这种方式而言，线程在处理完当前行之后，接下来要处理的行距离当前间隔为NUM\_PHTREAD，因此当数据规模很大的时候，会因为cache不能够容纳下足够的数据，或者由于CPU未能够及时的预取下一行数据，而导致cache miss，因此需要额外的访存开销。而块划分的方式就能够很好的弥补这一点，其原因是对于每个线程而言，他所需要处理的数据之间在内存上是连续的，因此有很好的cache优势，因此能够减小由于cache miss导致的额外访存开销。使用VTune工具对于这两种算法的L1 cache的命中率进行检测，如下表所示。块划分的命中率能够达到98\%，而循环划分的方式只有92\%左右，因此，对于块划分而言，由于其考虑到了cache特性，因此随着问题规模的增大，其性能并未明显受到访存开销的影响。而对于循环数据划分，则因为其划分方式会导致大量的cache miss，因此访存开销会极大影响其性能表现。这也正符合图中的变化趋势。

从负载均衡的角度出发，循环划分和动态数据划分的主要区别就在于能否充分利用各个线程的计算资源，尽可能减少同步等待所导致的额外开销。从图\ref{fig:ff6}中也可以看出，当问题规模较小的时候，动态划分方式的表现不如循环划分，这是由于动态划分在保证负载均衡的前提下，牺牲了线程调度的开销，由于每个线程不清楚自己具体的工作，因此会存在比较大的线程同步和线程切换的开销，这种额外开销在问题规模比较小的时候会格外显著。而当问题规模提升的时候，可以发现，动态划分方式的表现已经能够超越循环划分，负载均衡带来的收益已经抵消了线程调度的额外开销。根据VTune性能分析工具，观察这三种任务划分方式的CPU占用率，可以得到如下对比图。从途中可以看出，当动态数据划分的CPU占用率一直保持一个较高水平，并且相对比较均衡。而对比其余两种划分方式，由于其没有考虑负载均衡，因此在CPU占用率这个指标上，其波动十分明显，甚至会出现低于20%的占用率，这是对于计算资源的严重浪费。