__kernel void
transfer_global(__global __read_only float *input,
                 __global __write_only float *output,
                 int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < get_global_size(0) && y < get_global_size(1))
        output[y * N + x] = input[y * N + x];
}




__kernel void
transpose_global(__global __read_only float *input,
                 __global __write_only float *output,
                 int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < get_global_size(0) && y < get_global_size(1))
        output[x * N + y] = input[y * N + x];
}



__kernel void
transpose_local(__global __read_only float *input,
                __global __write_only float *output,
                __local float *buffer,
                int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int bufsize = get_local_size(0);

    if (x < get_global_size(0) && y < get_global_size(1))
        buffer[lx * bufsize + ly] = input[y * N + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < get_global_size(0) && y < get_global_size(1))
        output[y * N + x] = buffer[ly * bufsize + lx];
}

__kernel void
transpose_local_avoid_conflict(__global __read_only float *input,
                               __global __write_only float *output,
                               __local float *buffer,
                               int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int bufsize = get_local_size(0) + 1;

    if (x < get_global_size(0) && y < get_global_size(1))
        buffer[lx * bufsize + ly] = input[y * N + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < get_global_size(0) && y < get_global_size(1))
        output[y * N + x] = buffer[ly * bufsize + lx];
}
