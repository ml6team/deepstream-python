import ctypes
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')


def long_to_int(long):
    value = ctypes.c_int(long & 0xffffffff).value
    return value
