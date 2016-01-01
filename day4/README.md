## Timing results:

Current version: NVIDIA Jetson TK1
  * 7 zeros: 2.3-8.8s
  * 6 zeros: 0.8-2.2s
  * 5 zeros: 0.5-0.8s
  
28,500,000 hashes/sec

For a high-quality general purpose implementation, see http://hashcat.net/oclhashcat/ which is capable of 2,753,000 hashes/sec with a gtx580.  (Not sure why my number is higher.)

## Comparisons

Original solution: Python 2.7.9 using md5 library, Intel Core i5-4310 @ 2.6GHz
  * 6 zeros: 7.1s (average of 4 trials)
  * 5 zeros: 0.23s (average of 10 trials)
  
