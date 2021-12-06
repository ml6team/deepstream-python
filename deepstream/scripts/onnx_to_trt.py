import tensorrt as trt

onnx_path = "../data/sgies/osnet/osnet_x0_25_msmt17.onnx"

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print('ERROR: Failed to parse the ONNX file: {}'.format(onnx_path))
            for error in range(parser.num_errors):
                print(parser.get_error(error))
