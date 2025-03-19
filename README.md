# High Performance GEMM 

LD1 size, ways of associativity - 24 KiB, 12
L2 cache size, ways of associativity - 2048K, 12
Cache line size - 64 bytes

Parallelization: I chose to parallelize loop 2, as it gave the best performance 
boost. 

NC: I found that 11256 was large enough to maximize performance
KC: 22 / 14 * 1000 = 1571
MC: 15/16 * 2048/1571 * 1/8 = 152