/* -*- mode: C -*- */
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

typedef unsigned int uint32;

inline __device__ uint32 
leftRotate( uint32 x, uint32 c ) {
    return ( x << c ) | ( x >> (32 - c)); 
}


/* Assume trailer is 1 word only. */
inline __device__ uint32
getMessageWord( int g, uint32 *nonzeroWords, int nonzeroLen, uint32 trailer ) {
    if ( g < nonzeroLen ) {
        return nonzeroWords[g];
    } else if ( g == 14 ) {
        /* Length is a 64-bit quantity stored LSB first */
        return trailer;
    } else {
        return 0;
    }
}


#define SHIFT do {                                                         \
        uint32 dTemp = d;                                               \
        d = c;                                                          \
        c = b;                                                          \
        b = b + leftRotate( a + f + task->K[i] + getMessageWord( g, nonzeroWords, numWords, originalLen ), task->shiftTable[i] ); \
        a = dTemp;                                                      \
    } while ( false )
 
inline __device__ int
countLeadingZeros( uint32 a ) {
    return __clz( __byte_perm( a, 0, 0x0123 ) );
}

struct MD5Task {
    /* Fixed inputs */
    const int *shiftTable;
    const uint32 *K;

    const char *fixedPortion;
    int fixedLen;
    uint32 goalMask;

    /* Variable inputs */
    int startN;
    int batchSize;

    /* Output */
    int *goal;
};


__device__ bool md5Kernel_internal( struct MD5Task *task, uint32 n, int *localGoal );

/**
 * Calculate MD5 of "fixedPortion", a numeric suffix, and zero-padding.
 * Check for at least numZeros.
 */
__global__ void
md5Kernel( struct MD5Task task ) {
    /* The grid is a one-dimensional array of one-dimensional blocks. */
    uint32 startN = task.startN + blockDim.x * blockIdx.x + threadIdx.x;
    uint32 stride = blockDim.x * gridDim.x; 
    uint32 endN = startN + task.batchSize * stride;

    int * globalGoal = task.goal;
    __shared__ int localGoal;
    __shared__ char localPrefix[16];
    if ( threadIdx.x == 0 ) {
        localGoal = *globalGoal;
        /* FIXME: better to serialize this or distribute it? */
        memcpy( localPrefix, task.fixedPortion, task.fixedLen );
    }

    __shared__ int localShiftTable[64];
    __shared__ uint32 localKTable[64];
    /* Assuming at least 64 threads per block */
    if ( threadIdx.x < 64 ) {
        localShiftTable[threadIdx.x] = task.shiftTable[threadIdx.x];
        localKTable[threadIdx.x] = task.K[threadIdx.x];
    }

    /* Wait for shared memory to be filled before using it. */
    __syncthreads();

    task.shiftTable = localShiftTable;
    task.K = localKTable;
    task.fixedPortion = localPrefix;

    /* Check for end of run locally first.
       The thread that finds a solution also updates the global goal */
    for ( int n = startN; 
          n < endN && n < localGoal; 
          n += stride ) {
        md5Kernel_internal( &task, n, &localGoal );
    }    
}


__device__ bool
md5Kernel_internal( struct MD5Task *task, uint32 n, int *localGoal ) {
    /* Calculate log base 10 of n, and build a string that long */
    char numeric[16];
    int pos = 11;
    int val = n;
    for ( ; pos >= 0; --pos ) {
        numeric[pos] = '0' + ( val % 10 );
        val /= 10;
        if ( val == 0 ) break;
    }
    /* pos was the last character written. */
    int numberLen = 12 - pos;
    int numBytes = task->fixedLen + numberLen + 1;
    int numWords = ( numBytes + 3 ) / 4; /* Round up to the nearest 32-bit word. */
    uint32 nonzeroWords[ 15 ];
    nonzeroWords[numWords-1] = 0; /* Ensure zero bits at the end */

    /* Arrange the nonzero portion of the 512-bit chunk */
    char * nonzeroBytes = (char *)nonzeroWords;
    memcpy( nonzeroBytes, task->fixedPortion, task->fixedLen );
    memcpy( nonzeroBytes + task->fixedLen, numeric + pos, numberLen );
    nonzeroBytes[task->fixedLen + numberLen] = 0x80; /* 1 bit in MSB required */

    int originalLen = ( numBytes - 1 ) * 8;

    const uint32 a0 = 0x67452301;
    const uint32 b0 = 0xefcdab89;
    const uint32 c0 = 0x98badcfe;
    const uint32 d0 = 0x10325476;
    uint32 a = a0, b = b0, c = c0, d = d0;

    uint32 i;
    for ( i = 0; i <= 15; ++i ) {
        uint32 f = ( b & c ) | ( ~b & d );
        int g = i;
        SHIFT;
    }
    for ( i = 16; i <= 31; ++i ) {
        uint32 f = ( d & b ) | ( ~d & c );
        int g = (5 * i + 1) & 0xf;  /* mod 16 */
        SHIFT;
    }
    for ( i = 32; i <= 47; ++i ) {
        uint32 f = b ^ c ^ d;
        int g = ( 3 * i + 5 ) & 0xf; /* mod 16 */
        SHIFT;
    }

    for ( i = 48; i <= 63; ++i ) {
        uint32 f= c ^ (b | ~d);
        int g = ( 7 * i ) & 0xf;
        SHIFT;
    }

    a = a + a0;

    /* Needed for debug only */
    /*
    b = b + b0;
    c = c + c0;
    d = d + d0;
    */
    if ( ( a & task->goalMask ) == 0 ) {
        printf( "Goal: %d => %08x\n", n, a );
        *localGoal = n;
        atomicMin( task->goal, n ); 
        return true;
    } else {
        return false;
    }
}

int hostShiftTable[64] = {  
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 
};


uint32 hostKTable[64] = { 
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

#define checkCudaErrors(err) __checkCudaErrors( err, __FILE__, __LINE__ )

void __checkCudaErrors( cudaError_t err, const char * file, const int line ) {
    if ( err != cudaSuccess ) {
        fprintf( stderr, "Error %d (%s) at %s:%d\n",
                 err,
                 cudaGetErrorString( err ),
                 file,
                 line );
        exit( 1 );
    }
}

int 
usage() {
    fprintf( stderr, "Usage: day4 <secret key> <num zeros>\n" );
    fprintf( stderr, "  Number of zeros must be 1-8.\n" );
    exit( 2 );
}

int 
main( int argc, char *argv[] ) {
    if ( argc < 2 || argc > 3 ) usage();

    char *fixed = argv[1];
    char fixedLen = strlen( fixed );
    uint32 mask;
    if ( argc == 3 ) {
        int numZeros = atoi( argv[2] );
        if ( numZeros < 1 || numZeros > 8 ) usage();
        switch ( numZeros ) {
        case 1: mask = 0x000000f0; break;
        case 2: mask = 0x000000ff; break;
        case 3: mask = 0x0000f0ff; break;
        case 4: mask = 0x0000ffff; break;
        case 5: mask = 0x00f0ffff; break;
        case 6: mask = 0x00ffffff; break;
        case 7: mask = 0xf0ffffff; break;
        case 8: mask = 0xffffffff; break;
        }
    }

    int maxSearch = 1000000000;
    int goal = maxSearch;

    /* Device-side pointers to tables and input */
    int *deviceShiftTable;
    uint32 *deviceKTable;
    char *deviceInput;
    int *output;

    size_t tableSize = 64 * sizeof( uint32 );

    checkCudaErrors( cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeFourByte ) );
    
    checkCudaErrors( cudaMalloc( &deviceShiftTable, tableSize ) );
    checkCudaErrors( cudaMalloc( &deviceKTable, tableSize ) );
    checkCudaErrors( cudaMalloc( &deviceInput, fixedLen ) );
    checkCudaErrors( cudaMalloc( &output, sizeof( int ) ) );

    checkCudaErrors( cudaMemcpy( deviceShiftTable, hostShiftTable, tableSize, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( deviceKTable, hostKTable, tableSize, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( deviceInput, fixed, fixedLen, cudaMemcpyHostToDevice ) ); 
    checkCudaErrors( cudaMemcpy( output, &maxSearch, sizeof( int ), cudaMemcpyHostToDevice ) ); 
        
    /* My device has a maximum of 1024 threads per block and 
     * 2048 threads per multiprocessor.
     */
    int blockSize = 128;
    int numBlocks = 2048 / blockSize;
    /* 1B iterations / 2048 threads = 489000 per thread 
       Confusingly, large batch sizes make us slower.  */
    int batchSize = 200;
    int stride = blockSize * numBlocks * batchSize;

    struct MD5Task t;
    t.shiftTable = deviceShiftTable;
    t.K = deviceKTable;
    t.fixedPortion = deviceInput;
    t.fixedLen = fixedLen;
    t.goalMask = mask;
    t.goal = output;
    t.batchSize = batchSize;
    /* Checking for success *really* slows down the loop.  So we just launch some kernels that
     * might be unnecessary for small invocations.
     * FIXME: do the copy in parallel?
     */
    for ( int start = 0; start < maxSearch; start += stride ) {
        t.startN = start;
        md5Kernel<<<numBlocks,blockSize>>>( t );
        checkCudaErrors( cudaGetLastError() );
    }

    checkCudaErrors( cudaMemcpy( &goal, output, sizeof( int ), cudaMemcpyDeviceToHost ) ); 
    printf( "Answer: %d\n", goal );

    cudaFree( deviceShiftTable );
    cudaFree( deviceKTable );
    cudaFree( deviceInput );
    cudaFree( output );
    
    cudaDeviceReset();

    return 0;
}

 
