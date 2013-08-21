# PyOpenCl Degree Centrality by Alexander Sch\"afer
# based on examples by andreas kl\"ockner popencl development page, its mostly just a new context here
# speed up has to be taken with caution, it is mainly driven by the tresholding,
# no warranty, please check for errors

import pyopencl as cl
from time import time
import numpy as np
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt

ctx=cl.create_some_context()

queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

threshold=0.1;
from pyopencl.scan import GenericScanKernel
scan_kernel = GenericScanKernel(
        ctx, np.float32,
        arguments="__global float *ary, __global int segflag,__global float threshold",
        input_expr="(ary[i] < threshold) ? 0 : 1",
        scan_expr="across_seg_boundary ? b: (a+b)", neutral="0",is_segment_start_expr="(i)%segflag==0",
        output_statement="ary[i] = item+1;")

runs=15
cpu_time_array=np.zeros(runs-1)
gpu_time_array=np.zeros(runs-1)
gpu_total_time_array=np.zeros(runs-1)
for k in range(1,runs):
#####chose input size
    block_size = 16
    a_height = k*50*block_size
    a_width = k*50*block_size
    h_b_int = k*50*block_size

    c_width = a_width
    c_height = a_height

    h_a = np.random.rand(a_height*a_height).astype(np.float32)
    h_a_numpy = np.random.rand(a_height,a_height) ###numpy prefers matrices
    print h_a
    # transfer host -> device -----------------------------------------------------
    mf = cl.mem_flags

    t1 = time()

    a_gpu=cl.array.to_device(queue,h_a)

    push_time = time()-t1

    # warmup ----------------------------------------------------------------------
    for i in range(5):
        event = scan_kernel(a_gpu,h_b_int,threshold,queue=queue)
        event.wait()

    queue.finish()

    # actual benchmark ------------------------------------------------------------
    t1 = time()

    count =20
    for i in range(count):
        event = scan_kernel(a_gpu,h_b_int,threshold,queue=queue)

    event.wait()

    gpu_time = (time()-t1)/count

    # transfer device -> host -----------------------------------------------------
    t1 = time()
    #cl.enqueue_copy(queue, h_c, d_c_buf)
    pull_time = time()-t1

    # timing output ---------------------------------------------------------------
    gpu_total_time = gpu_time+push_time+pull_time

    print "GPU push+compute+pull total [s]:", gpu_total_time
    print "GPU push [s]:", push_time
    print "GPU pull [s]:", pull_time
    print "GPU compute (host-timed) [s]:", gpu_time
    print "GPU compute (event-timed) [s]: ", (event.profile.end-event.profile.start)*1e-9

    gflop = a_gpu.size * (a_width * 2.) / (1000**3.)
    gflops = gflop / gpu_time

    print
    print "GFlops/s:", gflops

    # cpu comparison --------------------------------------------------------------
    t1 = time()
    h_a_numpy[h_a_numpy<threshold] =0.0;
    h_a_numpy[h_a_numpy>threshold] =1.0;
    h_c_cpu = np.sum(h_a_numpy,axis=1)
    cpu_time = time()-t1
    print
    print "GPU:",(a_gpu)
    ##print "GPU-CPU:",(a_gpu-h_c_cpu)
    print
    print "CPU time (s)", cpu_time
    print

    print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
    print "GPU speedup (without transfer): ", cpu_time/gpu_time
    cpu_time_array[k-1]=cpu_time
    gpu_time_array[k-1]=gpu_time
    gpu_total_time_array[k-1]=gpu_total_time

plt.plot(range(1*50*block_size,runs*50*block_size,50*block_size),cpu_time_array,'b', label='CPU time',linewidth=2.0)
plt.hold(True)
plt.plot(range(1*50*block_size,runs*50*block_size,50*block_size),gpu_time_array,'r', label='GPU time',linewidth=2.0)
plt.plot(range(1*50*block_size,runs*50*block_size,50*block_size),gpu_total_time_array,'g', label='GPU total time',linewidth=2.0)
plt.xlabel('pixel squared')
plt.ylabel('time needed (s)')
plt.title('Performance Test for PyOpenCL vs. Numpy ')
plt.legend(loc=2)
plt.savefig('performance.png')
plt.show()

plt.plot(range(1*50*block_size,runs*50*block_size,50*block_size),cpu_time_array/gpu_time_array,'r', label='GPU speedup (without mem transfer)',linewidth=2.0)
plt.hold(True)
plt.plot(range(1*50*block_size,runs*50*block_size,50*block_size),cpu_time_array/gpu_total_time_array,'b', label="GPU speedup (with mem transfer)",linewidth=2.0)
plt.legend(loc=4)
plt.xlabel('pixel squared')
plt.ylabel('speed up')
#plt.ylim( (12, 34) )
plt.title('Speedup With and Without Memory Transfer')
plt.savefig('speedup.png')
plt.show()
