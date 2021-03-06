from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

if __name__ == '__main__':
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print('The queue is using the device: {}'.format(queue.device.name))

    program = cl.Program(context, open('transpose.cl').read()).build(options='')

    #
    # 2K x 2K image
    #
    image = np.arange(2000 * 2000).reshape((2000, 2000)).astype(np.float32)
    transposed = np.zeros_like(image)
    print("upper row:")
    print(image[0, :20].astype(int))

    #
    # allocate GPU-side memory
    #
    gpu_image = cl.Buffer(context, cl.mem_flags.READ_ONLY, image.size * 4)
    gpu_transposed = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, image.size * 4)

    # Workgroup sizes
    local_size = (16, 16)
    workgroup_size = local_size[0] * local_size[1]
    global_size = tuple([round_up(g, l) for g, l in zip(image.shape[::-1], local_size)])

    # copy image to GPU
    cl.enqueue_copy(queue, gpu_image, image, is_blocking=False)
    events = []
    for i in range(251):
        event = program.transfer_global(queue, global_size, local_size,
                                        gpu_image, gpu_transposed,
                                        np.int32(2000))
        events.append(event)

    cl.enqueue_copy(queue, transposed, gpu_transposed, is_blocking=True)

    print("upper row:")
    print(transposed[0, :20].astype(int))

    total_time = min((event.profile.end - event.profile.start) for event in events)
    print("best time, milliseconds: {}".format(total_time / 1e6))
