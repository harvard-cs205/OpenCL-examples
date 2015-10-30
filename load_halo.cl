__kernel void
load_halo(__global __read_only int *image,
          __global __read_only int *output,
          __local int *buffer,
          int w_img, int h_img,
          int w_buf, int h_buf,
          const int halo)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_group_size(0) + lx;

    int row, global_offset;

    if (idx_1D < w_buf)
        for (row = 0; row < h_buf; row++) {
            buffer[row * w_buf + idx_1D] = \
                FETCH(image, w_img, h_img,
                      buf_corner_x + idx_1D,
                      buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Processing code here...


    // write output
    output[y * w_img + x] = \
        buffer[(ly + halo) * w_buf + (lx + halo)];
}
