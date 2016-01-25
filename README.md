# inv_lift_test
Moving from CUDA 7.0 to 7.5 lead to an incorrect answer. The solution is for Visual Studio 2013. There is also a CMakeLists.txt as well. I've stripped out much of the original test harness to focus on the inv_lift function only for this reproducer.

Problem: 

I am working on a project to port ZFP (a compression library) to CUDA. With 7.0, the inv_xform function passed its test harness. However, when I updated to 7.5 inv_xform failed in the inv_lift function. After digging around more, I noticed that in CUDA 7.5 if I used the debug setting (-G), then I wouldn't get the error. That clued me into trying the volatile keyword. 

There are three inv_lift functions in this test case, kernel.cu: inv_lift_orig, inv_lift_fix1 and inv_lift_fix2. inv_lift_orig is the original code that failed. inv_lift_fix1 changed from a pointer increment to using array access. inv_lift_fix2 added a volatile keyword to local variables, which also fixed the problem. 

\__global\__ void gpuTest calls the inv_lift_orig in CUDA. The call to gpuTest is with one thread running on the GPU.

cpuTest calls inv_lift_orig directly.

Expected result: the value calculated on the GPU and the CPU are the same.
Actual result: the values are different on the CPU and GPU.

Tested: 
Alienware 17 r2 (i7-4720HQ and 980M) with Windows 10 and CUDA 7.0 and CUDA 7.5 (driver 353.90) with Visual Studio 2013, sm_52, compute_52 and debug mode off.
Acer Nitro v15 (i7-4720HQ and 860M) with Ubuntu 14.04.3 and CUDA 7.0. 
Also, i7-6700 and Quadro M4000 with Ubuntu 14.0.4.3 and  CUDA 7.5 (driver 352.63).


Output using CUDA 7.5 with a Quadro M4000 (Ubuntu 14.04.3):
```shell
mkim@hyper:~/inv_lift_test/DecodeTest$ /usr/local/cuda-7.5/bin/nvcc -arch=sm_52 kernel.cu 
kernel.cu(110): warning: variable "thread_cnt" was declared but never referenced

kernel.cu(110): warning: variable "thread_cnt" was declared but never referenced

mkim@hyper:~/inv_lift_test/DecodeTest$ ./a.out 
Borked (values should be the same):  at index 0 cpu value: -20 gpu value: 84
```

Ouput using CUDA 7.0 with a Quadro M4000 (Ubuntu 14.04.3):
```
mkim@hyper:~/inv_lift_test/DecodeTest$ /usr/local/cuda-7.0/bin/nvcc -arch=sm_52 kernel.cu 
kernel.cu(110): warning: variable "thread_cnt" was declared but never referenced

kernel.cu(110): warning: variable "thread_cnt" was declared but never referenced

mkim@hyper:~/inv_lift_test/DecodeTest$ ./a.out 
Finished correctly.
```
