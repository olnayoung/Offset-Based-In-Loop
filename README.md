## Offset-Based In-Loop Filtering With a Deep Network in HEVC
---
# So Yoon Lee; Yoonmo Yang; Dongsin Kim; Seunghyun Cho; Byung Tae Oh

In our recent [paper](https://ieeexplore.ieee.org/abstract/document/9272307), we proposed Offset-Based In-Looop Filtering.

With the great flexibility and performance of deep learning technology, there have been many attempts to replace existing functions inside video codecs such as High-Efficiency Video Coding (HEVC)
with deep-learning-based solutions. One of the most researched approaches is adopting a deep network as an image restoration filter to recover distorted compressed frames. In this paper, instead, we introduce
a novel idea for using a deep network, in which it chooses and transmits the side information according to the type of errors and contents, inspired by the sample adaptive offset filter in HEVC. A part of the
network computes the optimal offset values while another part estimates the type of error and contents simultaneously. The combination of two subnetworks can address the estimation of highly nonlinear and
complicated errors compared to conventional deep- learning-based schemes. Experimental results show that the proposed system yields an average bit-rate saving of 4.2% and 2.8% for the low-delay P and random
access modes, respectively, compared to the conventional HEVC. Moreover, the performance improvement is up to 6.3% and 3.9% for higher-resolution sequences.
