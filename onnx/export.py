import argparse
import pickle
import torch
import onnx
import onnxruntime as ort
import numpy as np
from torch import nn
from torch.nn import functional as F

# -------------------
# 与 train.py 中一致的模型定义（仅 input 输入，内部执行双线性上采样）
# -------------------

class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(QuantConv2d, self).__init__()
        if not type(kernel_size) in [list, tuple]:
            kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(
            (out_channels, in_channels, *kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 简化：忽略量化，仅普通卷积
        return F.conv2d(x, self.weight, bias=self.bias, padding='same')

class Net(nn.Module):
    def __init__(self, layers, rgb=False, quant_8=False, quant=False):
        super(Net, self).__init__()
        ch = 3 if rgb else 1
        ich = layers[0]
        self.cin = nn.Conv2d(ch, ich, 3, padding='same')
        nn.init.kaiming_normal_(self.cin.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.cin.bias)
        self.conv = nn.ModuleList()
        for och in layers[1:]:
            if quant_8:
                c = QuantConv2d(ich, och, 3, bias=False)
            else:
                c = nn.Conv2d(ich, och, 3, padding='same', bias=False)
            nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
            self.conv.append(c)
            ich = och
        self.cout = nn.Conv2d(ich, 4 * ch, 3, padding='same')
        nn.init.xavier_normal_(self.cout.weight)
        nn.init.zeros_(self.cout.bias)

    def forward(self, input):
        # 主流程：先特征提取
        x0 = F.relu(self.cin(input))
        for conv in self.conv:
            x0 = F.relu(conv(x0))
        out = self.cout(x0)
        out = F.pixel_shuffle(out, 2)
        # y 由 input 双线性上采样得到
        y = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.add(out, y)

# -------------------
# 脚本主流程
# -------------------

def load_pickle_state(pkl_path):
    with open(pkl_path, 'rb') as f:
        sd = pickle.load(f)
    rgb = bool(sd.get('rgb', False))
    quant_8 = bool(sd.get('quant-8', False))
    quant = bool(sd.get('quant', False))
    layers = sd.get('layers')
    state_dict = {}
    for k, v in sd.items():
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v)
    return state_dict, layers, rgb, quant_8, quant


def export_to_onnx(model, onnx_path, shape):
    # 将模型和输入转换为 fp16
    model = model.half()
    dummy = torch.randn(*shape).half()
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'h', 3: 'w'},
            'output': {0: 'batch', 2: 'h2', 3: 'w2'}
        }
    )
    print(f"✅ 导出 ONNX: {onnx_path} (fp16)")


def verify(onnx_path, shape):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 结构合法")
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name
    x = np.random.randn(*shape).astype(np.float16)
    out = sess.run(None, {inp_name: x})
    print("✅ 推理输出 shape:", [o.shape for o in out])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--shape', type=int, nargs=4, required=True,
                        help='batch,C,H,W for input')
    args = parser.parse_args()

    state_dict, layers, rgb, quant_8, quant = load_pickle_state(args.pkl)
    model = Net(layers, rgb=rgb, quant_8=quant_8, quant=quant)
    model.load_state_dict(state_dict)
    model.eval()

    export_to_onnx(model, args.onnx, tuple(args.shape))
    verify(args.onnx, tuple(args.shape))

if __name__ == '__main__':
    main()