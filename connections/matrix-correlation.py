# PyOpenCl Pearsons Correlation by Alexander Sch\"afer
# based on dot-product example by Eilif Mullers based on NVIDIA Matrix Multiplication http://developer.download.nvidia.com/compute/cuda/4_2/rel/sdk/website/OpenCL/html/samples.html and CUDA paper by Dar-Jen Chang et al, 2009, DOI 10.1109/SNPD.2009.34
# no warranty, please check for errors

KERNEL_CODE = """

// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width


#define Xs(j, i) Xs[i + j * BLOCK_SIZE]
#define Ys(j, i) Ys[i + j * BLOCK_SIZE]

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
matrixMul( __global float* C, __global float* A, int m,int n)
{
    __local float Xs[BLOCK_SIZE*BLOCK_SIZE];
    __local float Ys[BLOCK_SIZE*BLOCK_SIZE];
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    
    // Index of the first sub-matrix of A processed by the block
    int xBegin = WA * BLOCK_SIZE * bx;

    // Index of the last sub-matrix of A processed by the block
    int xEnd   = xBegin + WA - 1;

    // Step size used to iterate through the sub-matrices of X
    int xStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of X processed by the block
    int yBegin = WA*BLOCK_SIZE * by;

    // Step size used to iterate through the sub-matrices of X
    int yStep  = BLOCK_SIZE;

    // variables used later for pearson correlation computation
    float a1,a2,a3,a4,a5;
    float avgX,avgY,varX,varY,cov,rho;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int x = xBegin, y = yBegin; x <= xEnd; x+= xStep, y += yStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        Xs(tx, ty) = A[x + WA * ty + tx];
        Ys(ty, tx) = A[y + WA * ty + tx];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k){
            a1 += Xs(k,tx);
            a2 += Ys(ty, k);
            a3 += Xs(k, tx)*Xs(k, tx);
            a4 += Ys(ty, k)*Ys(ty, k);
            a5 += Xs(k, tx)*Ys(ty, k);
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of Xs and Ys in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    avgX=a1/m; 
    avgY=a2/m;
    varX=(a3-avgX*avgX*m)/(m -1);
    varY=(a4-avgY*avgY*m)/(m -1);
    cov=(a5-avgX*avgY*m)/(m-1);
    rho = cov/sqrt(varX*varY);
    // Write the block sub-matrix to device memory; each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = rho;

}

"""

import pyopencl as cl
from time import time
import numpy

block_size = 16

ctx = cl.create_some_context()

for dev in ctx.devices:
    assert dev.local_mem_size > 0

queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

#####chose input size
a_width = 10*block_size
a_height = 1000*block_size

c_width = a_width
c_height = a_height

h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
h_c = numpy.empty((a_height, a_height)).astype(numpy.float32)
print h_a.shape
m=a_width;
n=a_height;

kernel_params = {"block_size": block_size, "w_a":a_width, "h_a":a_height}

if "NVIDIA" in queue.device.vendor:
    options = "-cl-mad-enable -cl-fast-relaxed-math"
else:
    options = ""
prg = cl.Program(ctx, KERNEL_CODE % kernel_params,
        ).build(options=options)
kernel = prg.matrixMul


assert a_width % block_size == 0
assert a_height % block_size == 0

# transfer host -> device -----------------------------------------------------
mf = cl.mem_flags

t1 = time()

d_a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=h_c.nbytes)

push_time = time()-t1

# warmup ----------------------------------------------------------------------
for i in range(5):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size), d_c_buf, d_a_buf, numpy.uint32(m),numpy.uint32(n))
    event.wait()

queue.finish()

# actual benchmark ------------------------------------------------------------
t1 = time()

count = 20
for i in range(count):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size),d_c_buf, d_a_buf,numpy.uint32(m),numpy.uint32(n))

event.wait()

gpu_time = (time()-t1)/count

# transfer device -> host -----------------------------------------------------
t1 = time()
cl.enqueue_copy(queue, h_c, d_c_buf)
pull_time = time()-t1

# timing output ---------------------------------------------------------------
gpu_total_time = gpu_time+push_time+pull_time

print "GPU push+compute+pull total [s]:", gpu_total_time
print "GPU push [s]:", push_time
print "GPU pull [s]:", pull_time
print "GPU compute (host-timed) [s]:", gpu_time
print "GPU compute (event-timed) [s]: ", (event.profile.end-event.profile.start)*1e-9

gflop = h_c.size * (a_width * 2.) / (1000**3.)
gflops = gflop / gpu_time

print
print "GFlops/s:", gflops

# cpu comparison --------------------------------------------------------------
t1 = time()
h_c_cpu = numpy.corrcoef(h_a)
cpu_time = time()-t1

print
print "GPU-CPU:",(h_c-h_c_cpu)
print
print "CPU time (s)", cpu_time
print

print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
print "GPU speedup (without transfer): ", cpu_time/gpu_time

