## Timing results:

Current version: NVIDIA Jetson TK1
  * 8 zeros (maximum 10^9 searches): 13.2s+
  * 7 zeros: 3.5s
  * 6 zeros: 0.3s
  * 5 zeros: 0.2s
 
75,000,000 hashes per second.

For a high-quality general purpose implementation, see http://hashcat.net/oclhashcat/ which is capable of 2,753,000 hashes/sec with a gtx580.  (Not sure why my number is higher.)

## Comparisons

Original solution: Python 2.7.9 using md5 library, Intel Core i5-4310 @ 2.6GHz
  * 6 zeros: 7.1s (average of 4 trials)
  * 5 zeros: 0.2s (average of 10 trials)
  
