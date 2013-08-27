# PyOpenCl Degree Centrality by Alexander Sch\"afer
# based on examples by andreas kl\"ockner popencl development page
# speed up has to be taken with caution, it is mainly driven by the tresholding,
# no warranty, please check for errors

import pyopencl as cl
from time import time
import numpy as np
from pyopencl.scan import GenericScanKernel

def matrix_deg_centrality(h_a,threshold,a_height):
    ### h_a is the input matrix in array form, so shape=(rowsxcolumns,1)
    ### assumes that the connectivity matrix is symmetric
    ### threshold is the threshold applied to the connectivity matrix
    ### a_height is the number of columns or row of the input matrix


    block_size = 16
    a_width = a_height###assumes symmetric matrix
    h_b_int = a_height
    c_width = a_width
    c_height = a_height
    h_result=np.empty(a_height).astype(np.float32);


    ctx=cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    from pyopencl.scan import GenericScanKernel
    scan_kernel = GenericScanKernel(
            ctx, np.float32,
            arguments="__global float *ary,__global float *out, __global int segflag,__global float threshold",
            input_expr="(ary[i] < threshold) ? 0 : 1",
            scan_expr="across_seg_boundary ? b: (a+b)", neutral="0",is_segment_start_expr="(i)%segflag==0",
            output_statement="(i+1)%segflag==0 ? (out[i/segflag] = item,ary[i] = item) : (ary[i] = item);")


    mf = cl.mem_flags
    a_gpu=cl.array.to_device(queue,h_a)
    result_gpu=cl.array.to_device(queue,h_result)
    event = scan_kernel(a_gpu,result_gpu,h_b_int,threshold,queue=queue)
    gpu_centrality= result_gpu.get(); ##check if everything is correct
    return gpu_centrality

def test_deg_centrality():
    ###test function
    a_height=10000;
    matrix=np.random.rand(a_height*a_height).astype(np.float32);
    threshold=0.5;
    test=matrix_deg_centrality(matrix,threshold,a_height);
    print test
