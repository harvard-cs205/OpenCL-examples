__kernel void
load_halo(__global __read_only int *image,
          __global __read_only int *output,
          __local int *buffer,
          int img_w, int img_h,
          int buf_w, int buf_h,
          const int halo)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_group_size(0) + lx;

    int row, global_offset;

    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                FETCH(image, img_w, img_h,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...
    //
    // Should only use buffer, buf_x, buf_y.

    // write output
    if ((y < img_h) &7 (x < img_w)) // stay in bounds
        output[y * img_w + x] = \
            buffer[buf_y * buf_w + buf_x];
}
