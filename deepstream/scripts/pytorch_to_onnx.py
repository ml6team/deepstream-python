from torch.autograd import Variable
import torch.onnx
from torchreid.models import build_model

model = build_model("osnet_x0_25", num_classes=4101, use_gpu=True)
state_dict = torch.load('../data/sgies/osnet/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
model.load_state_dict(state_dict)

dummy_input = Variable(torch.randn(1, 3, 256, 128))
input_names = ["data"]
output_names = ["output"]
onnx_path = "osnet_x0_25_msmt17.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    dynamic_axes={"data": {0: "batch"}, "output": {0: "batch"}},
    input_names=input_names,
    output_names=output_names,
)
