__kernel void
transfer_global(__global __read_only float *input,
                 __global __write_only float *output,
                 int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < N && y < N)
        output[y * N + x] = input[y * N + x];
}




__kernel void
transpose_global(__global __read_only float *input,
                 __global __write_only float *output,
                 int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < N && y < N)
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

    if (x < N && y < N)
        buffer[lx * bufsize + ly] = input[y * N + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int base = (x - lx) * N + (y - ly);
    if (x < N && y < N)
        output[base + ly * N + lx] = buffer[ly * bufsize + lx];
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

    if (x < N && y < N)
        buffer[lx * bufsize + ly] = input[y * N + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int base = (x - lx) * N + (y - ly);
    if (x < N && y < N)
        output[base + ly * N + lx] = buffer[ly * bufsize + lx];
}




__kernel void
transfer_global_measure_occupancy(__global __read_only float *input,
                                  __global __write_only float *output,
                                  __global int *active,
                                  __global int *max_active,
                                  int N)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (get_local_id(0) + get_local_id(1) == 0) {
        int old = atomic_inc(active);
        atomic_max(max_active, old + 1);
    }

    if (x < N && y < N)
        output[y * N + x] = input[y * N + x];

    if (get_local_id(0) + get_local_id(1) == 0)
        atomic_dec(active);
}
