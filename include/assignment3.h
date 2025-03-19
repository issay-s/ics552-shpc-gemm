#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>


#define dabs( x ) ( (x) < 0 ? -(x) : x )
#define max(x, y) (((x) > (y)) ? (x) : (y))

#define MR 8
#define NR 6
#define KC 256
#define MC 960
#define NC 11256
#define UNROLLING 2

#include "blis.h"

#include "src.h"
#include "test.h"                                                        
#include "util.h"

