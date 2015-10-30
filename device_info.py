from __future__ import division
import pyopencl as cl

# List our platforms
platforms = cl.get_platforms()
print 'The platforms detected are:'
print '---------------------------'
for platform in platforms:
    print platform.name, platform.vendor, 'version:', platform.version

# List devices in each platform
for platform in platforms:
    print 'The devices detected on platform', platform.name, 'are:'
    print '---------------------------'
    for device in platform.get_devices():
        print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
        print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
        print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
        print 'Maximum work group size', device.max_work_group_size
        print 'Maximum work item dimensions', device.max_work_item_dimensions
        print 'Maximum work item size', device.max_work_item_sizes
        print '---------------------------'
