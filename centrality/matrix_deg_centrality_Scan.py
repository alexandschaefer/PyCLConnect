# PyOpenCl Degree Centrality by Alexander Sch\"afer
# based on examples by andreas kl\"ockner popencl development page
# speed up has to be taken with caution, it is mainly driven by the tresholding,
# no warranty, please check for errors

import pyopencl as cl
from time import time
import numpy as np
from pyopencl.scan import GenericScanKernel


#####chose input size
block_size = 16
a_height = 10000#800*block_size
a_width = 10000#800*block_size
h_b_int = 10000#800*block_size

c_width = a_width
c_height = a_height

h_a = np.random.rand(a_height*a_height).astype(np.float32)
h_result = np.random.rand(a_height).astype(np.float32)
h_a_numpy = np.random.rand(a_height,a_height) ###numpy prefers matrices
print h_a



ctx=cl.create_some_context()

queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

threshold=0.5;
from pyopencl.scan import GenericScanKernel
scan_kernel = GenericScanKernel(
        ctx, np.float32,
        arguments="__global float *ary,__global float *out, __global int segflag,__global float threshold",
        input_expr="(ary[i] < threshold) ? 0 : 1",
        scan_expr="across_seg_boundary ? b: (a+b)", neutral="0",is_segment_start_expr="(i)%segflag==0",
        output_statement="(i+1)%segflag==0 ? (out[i/segflag] = item,ary[i] = item) : (ary[i] = item);")


# transfer host -> device -----------------------------------------------------
mf = cl.mem_flags

t1 = time()

a_gpu=cl.array.to_device(queue,h_a)
result_gpu=cl.array.to_device(queue,h_result)
push_time = time()-t1

# warmup ----------------------------------------------------------------------
for i in range(0):
    event = scan_kernel(a_gpu,result_gpu,h_b_int,threshold,queue=queue)
    event.wait()

queue.finish()

# actual benchmark ------------------------------------------------------------
t1 = time()

count =1
for i in range(count):
    event = scan_kernel(a_gpu,result_gpu,h_b_int,threshold,queue=queue)

event.wait()

gpu_time = (time()-t1)/count

# transfer device -> host -----------------------------------------------------
t1 = time()
#cl.enqueue_copy(queue, h_c, d_c_buf)

gpu_centrality= result_gpu.get(); ##check if everything is correct

pull_time = time()-t1
result= a_gpu.get();
centrality=result[h_b_int-1:h_b_int*h_b_int:h_b_int] 
print centrality
print gpu_centrality
print result
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
##print "GPU-CPU:",(a_gpu-h_c_cpu)
print
print "CPU time (s)", cpu_time
print

print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
print "GPU speedup (without transfer): ", cpu_time/gpu_time

