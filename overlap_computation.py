import pyopencl as cl
import numpy as np

platforms = cl.get_platforms()
devices = platforms[0].get_devices()
context = cl.create_some_context()

src = """
kernel void smooth(global float* input,
                   global float* output,
                   int N)
{
    int i = get_global_id(0);

    if (i < N)
        output[i] = (0.5 * input[i] +
                     0.25 * input[max(0, i - 1)] +
                     0.25 * input[min(N - 1, i + 1)]);
}
"""
program = cl.Program(context, src).build()

copy_in_queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
execute_queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
copy_out_queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

serialize = False
if serialize:
    execute_queue = copy_out_queue = copy_in_queue

input_values = [np.random.uniform(0, 1, 1000000).astype(np.float32) for idx in range(20)]
output_values = [np.empty_like(i) for i in input_values]
clmem = cl.mem_flags
gpu_in_bufs = [cl.Buffer(context, clmem.READ_ONLY | clmem.COPY_HOST_PTR, hostbuf=i) for i in input_values]
gpu_out_bufs = [cl.Buffer(context, clmem.READ_WRITE, 4 * i.size) for i in input_values]

global_size = input_values[0].shape

in_evs = []
ex_evs = []
out_evs = []
copy_out_event = None
for idx in range(len(input_values)):
    in_wait = []
    if serialize and (copy_out_event is not None):
        in_wait = [copy_out_event]
    copy_in_event = cl.enqueue_copy(copy_in_queue, gpu_in_bufs[idx], input_values[idx],
                                    is_blocking=False, wait_for=in_wait)
    copy_in_event = cl.enqueue_copy(copy_in_queue, gpu_in_bufs[idx], input_values[idx], is_blocking=False)
    execute_event = program.smooth(execute_queue, global_size, None,  # let OpenCL choose the workgroup size
                                   gpu_in_bufs[idx], gpu_out_bufs[idx], np.int32(input_values[idx].size),
                                   wait_for=[copy_in_event])
    copy_out_event = cl.enqueue_copy(copy_out_queue, output_values[idx],
                                     gpu_out_bufs[idx], wait_for=[execute_event],
                                     is_blocking=False)
    in_evs.append(copy_in_event)
    ex_evs.append(execute_event)
    out_evs.append(copy_out_event)

base = None
for ev_in, ev_ex, ev_out in zip(in_evs, ex_evs, out_evs):
    ev_in.wait()
    if base is None:
        base = ev_in.profile.start
    def wait_and_print(ev, name):
        ev.wait()
        print("{}: {} to {}  \t({} ns)".format(name,
                                    ev.profile.start - base,
                                    ev.profile.end - base,
                                    ev.profile.end - ev.profile.start))
    wait_and_print(ev_in,  "Copy In ")
    wait_and_print(ev_ex,  "Execute ")
    wait_and_print(ev_out, "Copy Out")
    print("")
