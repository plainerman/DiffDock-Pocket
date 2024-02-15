import numpy as np
import torch
from e3nn import o3

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


class FasterTensorProduct(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, **kwargs):
        super().__init__()
        #for ir in in_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order in irreps are not supported"
        #for ir in out_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order out irreps are not supported"
        assert o3.Irreps(sh_irreps) == o3.Irreps('1x0e+1x1o'), "sh_irreps don't look like 1st order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        in_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        out_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        for (m, ir) in self.in_irreps: in_muls[str(ir)] = m
        for (m, ir) in self.out_irreps: out_muls[str(ir)] = m

        self.weight_shapes = {
            '0e': (in_muls['0e'] + in_muls['1o'], out_muls['0e']),
            '1o': (in_muls['0e'] + in_muls['1o'] + in_muls['1e'], out_muls['1o']),
            '1e': (in_muls['1o'] + in_muls['1e'] + in_muls['0o'], out_muls['1e']),
            '0o': (in_muls['1e'] + in_muls['0o'], out_muls['0o'])
        }
        self.weight_numel = sum(a * b for (a, b) in self.weight_shapes.values())

    def forward(self, in_, sh, weight):
        in_dict, out_dict = {}, {'0e': [], '1o': [], '1e': [], '0o': []}
        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]
            if ir[0] == 1: in_dict[str(ir)] = in_dict[str(ir)].reshape(list(in_dict[str(ir)].shape)[:-1] + [-1, 3])
        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]
        if '0e' in in_dict:
            out_dict['0e'].append(in_dict['0e'] * sh_0e.unsqueeze(-1))
            out_dict['1o'].append(in_dict['0e'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if '1o' in in_dict:
            out_dict['0e'].append((in_dict['1o'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
            out_dict['1o'].append(in_dict['1o'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['1e'].append(torch.linalg.cross(in_dict['1o'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
        if '1e' in in_dict:
            out_dict['1o'].append(torch.linalg.cross(in_dict['1e'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
            out_dict['1e'].append(in_dict['1e'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['0o'].append((in_dict['1e'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
        if '0o' in in_dict:
            out_dict['1e'].append(in_dict['0o'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict['0o'].append(in_dict['0o'] * sh_0e.unsqueeze(-1))

        weight_dict = {}
        start = 0
        for key in self.weight_shapes:
            in_, out = self.weight_shapes[key]
            weight_dict[key] = weight[..., start:start + in_ * out].reshape(
                list(weight.shape)[:-1] + [in_, out]) / np.sqrt(in_)
            start += in_ * out

        if out_dict['0e']:
            out_dict['0e'] = torch.cat(out_dict['0e'], dim=-1)
            out_dict['0e'] = torch.matmul(out_dict['0e'].unsqueeze(-2), weight_dict['0e']).squeeze(-2)

        if out_dict['1o']:
            out_dict['1o'] = torch.cat(out_dict['1o'], dim=-2)
            out_dict['1o'] = (out_dict['1o'].unsqueeze(-2) * weight_dict['1o'].unsqueeze(-1)).sum(-3)
            out_dict['1o'] = out_dict['1o'].reshape(list(out_dict['1o'].shape)[:-2] + [-1])

        if out_dict['1e']:
            out_dict['1e'] = torch.cat(out_dict['1e'], dim=-2)
            out_dict['1e'] = (out_dict['1e'].unsqueeze(-2) * weight_dict['1e'].unsqueeze(-1)).sum(-3)
            out_dict['1e'] = out_dict['1e'].reshape(list(out_dict['1e'].shape)[:-2] + [-1])

        if out_dict['0o']:
            out_dict['0o'] = torch.cat(out_dict['0o'], dim=-1)
            # out_dict['0o'] = (out_dict['0o'].unsqueeze(-1) * weight_dict['0o']).sum(-2)
            out_dict['0o'] = torch.matmul(out_dict['0o'].unsqueeze(-2), weight_dict['0o']).squeeze(-2)

        out = []
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)])
        return torch.cat(out, dim=-1)
