/* -*- mode: C -*- */
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

typedef unsigned int uint32;

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

// Sized for up to 10 digits + 1 byte for '1' bit + 9 fixed chars
struct Header {
    uint32 word[5];
    uint32 originalLen;
};

/* logical left-shifting every digit produces, in little-endian:
 *                 1     x    x   x
 *                 10    1    x   x
 *                 100   10   1   x
 *                 1000  100  10  1
 *                 10000 1000 100 10  1  x  x  x 
 * 
 * So we just have to get the word order right, i.e., 
 * MSB of word N becomes LSB of word N+1
 */

inline __device__ struct Header
shiftHeaderLeft( struct Header in ) {
    struct Header out;
    out.word[0] = ( in.word[0] << 8 );    
    out.word[1] = ( in.word[0] >> 24 ) | ( in.word[1] << 8 );
    out.word[2] = ( in.word[1] >> 24 ) | ( in.word[2] << 8 );
    out.word[3] = ( in.word[2] >> 24 ) | ( in.word[3] << 8 );
    out.word[4] = ( in.word[3] >> 24 ) | ( in.word[4] << 8 );
    return out;
}

/* __byte_perm orders inputs from LSB to MSB of word A, then word B. 
   Output is input[s[0:3]] ## input[s[4:7]] ## input[s[8:11]] ## input[s[12:15]] */


__device__ struct Header
constructHeaderWithShift( const char * fixedPortion, int fixedLen, int n ) {
    /* fixedLen should be 0-3 */
    struct Header number;
    int digitLen = 0;

    number.word[0] = 0x80; /* This is the mandatory 1 bit that will appear at the end */                                               
    number.word[1] = 0;
    number.word[2] = 0;
    number.word[3] = 0;
    number.word[4] = 0;

    do {
        number = shiftHeaderLeft( number );
        number.word[0] |= ( '0' + ( n % 10 ) );
        n /= 10;
        digitLen += 1;
    } while ( n > 0 );

    for ( int k = 0; k < fixedLen; ++k ) {
        /* One more shift for the fixed portion */
        number = shiftHeaderLeft( number );
    }
    if ( fixedLen == 1 ) {
        number.word[0] |= (uint32)( fixedPortion[0] );
    } else if ( fixedLen == 2 ) {
        number.word[0] |= (( *(uint32 *)( fixedPortion )) & 0xffff );
    } else if ( fixedLen == 3 ) {
        number.word[0] |= (( *(uint32 *)( fixedPortion )) & 0xffffff );
    }

    number.originalLen = ( digitLen + fixedLen ) * 8;
    return number;
}

__device__ struct Header 
constructHeader( struct MD5Task *task, int n ) {
    /* Every element of the block has the same fixed length,
       and most have the same # of digits in N as well. 
       Not fully implemented at the moment. */
    if ( task->fixedLen < 4 ) {
        return constructHeaderWithShift( task->fixedPortion, task->fixedLen, n );
    } else if ( task->fixedLen < 8 ) {
        struct Header tail = constructHeaderWithShift( task->fixedPortion + 4, task->fixedLen - 4, n );
        struct Header ret;
        ret.word[0] = *(uint32 *)( task->fixedPortion );
        ret.word[1] = tail.word[0];
        ret.word[2] = tail.word[1];
        ret.word[3] = tail.word[2];
        ret.word[4] = tail.word[3];
        ret.originalLen = tail.originalLen + 32; /* in bits, not bytes */
        return ret;
    } else {
        struct Header tail = constructHeaderWithShift( task->fixedPortion + 8, task->fixedLen - 8, n );
        struct Header ret;
        ret.word[0] = *(uint32 *)( task->fixedPortion );
        ret.word[1] = *(uint32 *)( task->fixedPortion + 4 );
        ret.word[2] = tail.word[0];
        ret.word[3] = tail.word[1];
        ret.word[4] = tail.word[2];
        ret.originalLen = tail.originalLen + 64; /* in bits, not bytes */
        return ret;
    }
    /* FIXME: check for out-of-bounds behavior */
}

/* Assume trailer is 1 word only. */
inline __device__ uint32
getMessageWord( struct Header h, int i ) {
    /* I'm hoping that not using [] will result in each word
     * being assigned to a register, rather than using memory acccess.
     */
    if ( i == 0 ) {
        return h.word[0];
    } else if ( i == 1 ) {
        return h.word[1];
    } else if ( i == 2 ) {
        return h.word[2];
    } else if ( i == 3 ) {
        return h.word[3];
    } else if ( i == 4 ) {
        return h.word[4];
    } else if ( i == 14 ) {
        /* Length is a 64-bit quantity stored LSB first */
        return h.originalLen;
    } else {
        return 0;
    }
}

inline __device__ uint32 
leftRotate( uint32 x, uint32 c ) {
    return ( x << c ) | ( x >> (32 - c)); 
}

#define SHIFT do {                                                         \
        uint32 dTemp = d;                                               \
        d = c;                                                          \
        c = b;                                                          \
        b = b + leftRotate( a + f + task->K[i] + getMessageWord( md5Header, g ), task->shiftTable[i] ); \
        a = dTemp;                                                      \
    } while ( false )
 
__device__ bool
md5Kernel_internal( struct MD5Task *task, uint32 n, int *localGoal ) {
    struct Header md5Header = constructHeader( task, n );
 
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

    //int maxSearch = 100;
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
    checkCudaErrors( cudaMalloc( &deviceInput, fixedLen + 16 ) ); /* Allocate extra space */
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

 
