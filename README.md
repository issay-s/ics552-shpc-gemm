**SHPC DGEMM: Optimized Matrix Multiplication**  
This project implements a high-performance double-precision general matrix multiplication (DGEMM) routine using AVX intrinsics, OpenMP for parallelism, and cache-aware blocking and packing.  
Cache-optimization/blocking allows this implementation to **match or outperform BLIS Library (industry standard)** and paralelization allows us to perform **multiple times faster than BLAS Library** on the same machine. 
  ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
**Overview**  
Blocked and packed matrix multiplication: C = A * B + C  
Vectorized with AVX (256-bit) intrinsics  
Multithreaded using OpenMP  
Handles arbitrary matrix sizes with edge-case logic  
Supports general memory layouts via row and column strides  
  
**Implementation Details**
Blocking: Improves cache performance using hierarchical block sizes (MC, NC, KC)  
Packing: Rearranges panels of A and B into contiguous buffers  
Microkernel: Performs 8x6 multiplication using fused multiply-add (FMA)  
Parallelism: Outer column loop is parallelized using OpenMP  
  
**Function structure**  
shpc_dgemm(m, n, k,  
           A, rsA, csA,  
           B, rsB, csB,  
           C, rsC, csC);  
A, B, C: pointers to matrices  
rs*, cs*: row and column strides  

**Other specifications**  
LD1 size, ways of associativity - 24 KiB, 12  
L2 cache size, ways of associativity - 2048K, 12  
Cache line size - 64 bytes  
  
Parallelization: I chose to parallelize loop 2, as it gave the best performance   
boost.   
  
NC: I found that 11256 was large enough to maximize performance  
KC: 22 / 14 * 1000 = 1571  
MC: 15/16 * 2048/1571 * 1/8 = 152  
