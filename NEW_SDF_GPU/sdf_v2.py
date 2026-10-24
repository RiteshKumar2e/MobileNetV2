"""SDF-YOLO v2: SDF-YOLO combined with the components of APG-YOLO (Liao et al.,
The Visual Computer 2026) -- PKI block, AIFI, GFPN-style neck -- built so that every
YOLO11 layer it keeps still starts from COCO weights.

Differences from APG-YOLO, all for transfer learning on small defect datasets:
  * PKI blocks are ADDED after the pretrained C3k2 stages as zero-init residuals
    (out = x + gamma * PKI(x), gamma = 0), instead of replacing C3k2. The network
    starts exactly as COCO YOLO11 and the PKI branches grow in during training.
  * P5 = pretrained SPPF followed by AIFI (APG replaces SPPF by AIFI).
  * Neck = YOLO11's pretrained top-down FPN, then a GFPN-style bottom-up path:
    every level concatenates the downsampled lower level, the same-level top-down
    feature, the raw backbone feature (dense skip) and the upsampled higher level
    (queen fusion). P3 also takes the stride-4 backbone feature. SDF's multi-scale
    DWConv block and CAA stay on the outputs.
  * Optional P2 detection level (--p2-head) for tiny defects.

Weights are transferred by an explicit layer-index map (layers move when blocks are
inserted, so ultralytics' name-based load() would miss them).
"""
import torch
import torch.nn as nn
import yaml
import ultralytics.nn.tasks as _tasks
from ultralytics.nn.modules import Conv
from ultralytics.utils.ops import make_divisible


class CAAGate(nn.Module):
    """CAA attention map A = sigmoid(conv1x1(DW kx1(DW 1xk(conv1x1(avgpool(x))))))."""

    def __init__(self, c, k=11):
        super().__init__()
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(c, c, 1)
        self.h = nn.Conv2d(c, c, (1, k), 1, (0, k // 2), groups=c)
        self.v = nn.Conv2d(c, c, (k, 1), 1, (k // 2, 0), groups=c)
        self.conv2 = Conv(c, c, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2(self.v(self.h(self.conv1(self.pool(x))))))


class PKI(nn.Module):
    """Poly-Kernel Inception block (APG-YOLO eqs. 1-6) as a zero-init residual:
    L = Conv3x3(x); P = Conv1x1(L + sum_m DWConv_km(L)); A = CAA(x);
    out = x + gamma * Conv1x1(P + P * A)."""

    def __init__(self, c1, kernels=(3, 5, 7, 9, 11), caa_k=11, e=0.5):
        super().__init__()
        c_ = make_divisible(c1 * e, 8)
        self.pre = Conv(c1, c_, 1)
        self.local = Conv(c_, c_, 3)
        self.dw = nn.ModuleList(nn.Conv2d(c_, c_, k, 1, k // 2, groups=c_, bias=False) for k in kernels)
        self.pw = Conv(c_, c_, 1)
        self.caa = CAAGate(c_, caa_k)
        self.out = Conv(c_, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(1, c1, 1, 1))

    def forward(self, x):
        y = self.pre(x)
        loc = self.local(y)
        p = self.pw(loc + sum(dw(loc) for dw in self.dw))
        p = p + p * self.caa(y)
        return x + self.gamma * self.out(p)


_tasks.PKI = PKI


class _B:
    def __init__(self):
        self.layers, self.src = [], {}

    def add(self, frm, module, args, src=None):
        self.layers.append([frm, 1, module, list(args)])
        i = len(self.layers) - 1
        if src is not None:
            self.src[i] = src
        return i


def build_sdf_v2_yaml(nc, scale, scales, out_path, pki_kernels=(3, 5, 7, 9, 11),
                      pki_levels=(4, 6, 8), msdw_kernels=(3, 5, 7, 9), caa_kernels=(11, 9, 7),
                      aifi=True, p2_head=False):
    """Returns (yaml path, {sdf layer index: yolo11 layer index}, n detect levels) -- the
    map drives the weight transfer. yolo11 indices: backbone 0-10, top-down FPN 11-16,
    Detect 23."""
    depth, width, max_ch = scales[scale]

    def sc(c):
        return make_divisible(min(c, max_ch) * width, 8)

    b = _B()

    def pki(i, c, lvl):
        if lvl not in pki_levels:
            return i
        return b.add(i, "PKI", [sc(c), list(pki_kernels), 11])

    def caa(i, c, k):
        return b.add(i, "CAA", [sc(c), k, True])

    b0 = b.add(-1, "Conv", [64, 3, 2], 0)
    b1 = b.add(b0, "Conv", [128, 3, 2], 1)
    b2 = b.add(b1, "C3k2", [256, False, 0.25], 2)
    b2 = pki(b2, 256, 2)
    b3 = b.add(b2, "Conv", [256, 3, 2], 3)
    b4 = b.add(b3, "C3k2", [512, False, 0.25], 4)
    b4 = pki(b4, 512, 4)
    b5 = b.add(b4, "Conv", [512, 3, 2], 5)
    b6 = b.add(b5, "C3k2", [512, True], 6)
    b6 = pki(b6, 512, 6)
    b7 = b.add(b6, "Conv", [1024, 3, 2], 7)
    b8 = b.add(b7, "C3k2", [1024, True], 8)
    b8 = pki(b8, 1024, 8)
    b9 = b.add(b8, "SPPF", [1024, 5], 9)
    if aifi:
        b9 = b.add(b9, "AIFI", [1024, 8])
    b10 = b.add(b9, "C2PSA", [1024], 10)
    n_backbone = len(b.layers)

    # pretrained top-down FPN (same shapes as yolo11 layers 11-16)
    u = b.add(b10, "nn.Upsample", [None, 2, "nearest"], 11)
    c = b.add([u, b6], "Concat", [1], 12)
    t4 = b.add(c, "C3k2", [512, False], 13)
    u = b.add(t4, "nn.Upsample", [None, 2, "nearest"], 14)
    c = b.add([u, b4], "Concat", [1], 15)
    t3 = b.add(c, "C3k2", [256, False], 16)

    outs = []
    low = b.add(b2, "Conv", [256, 3, 2])             # stride-4 detail -> P3
    if p2_head:
        u = b.add(t3, "nn.Upsample", [None, 2, "nearest"])
        c = b.add([u, b2], "Concat", [1])
        p2 = b.add(c, "C3k2", [256, False])
        p2 = caa(p2, 256, 13)
        outs.append(p2)
        low = b.add(p2, "Conv", [256, 3, 2])

    # P3: queen fusion (lower level, same level, upsampled P4) + MSDW + CAA
    u4 = b.add(t4, "nn.Upsample", [None, 2, "nearest"])
    c = b.add([t3, u4, low], "Concat", [1])
    p3 = b.add(c, "C3k2", [256, False])
    if msdw_kernels:
        x = b.add(p3, "Conv", [256, 3, 1])
        br = [b.add(x, "DWConv", [256, k, 1]) for k in msdw_kernels]
        c = b.add([x, *br], "Concat", [1])
        p3 = b.add(c, "Conv", [256, 1, 1])
    p3 = caa(p3, 256, caa_kernels[0])
    outs.append(p3)

    # P4: down(P3) + top-down P4 + raw backbone P4 + upsampled P5
    d = b.add(p3, "Conv", [512, 3, 2])
    u5 = b.add(b10, "nn.Upsample", [None, 2, "nearest"])
    c = b.add([d, t4, b6, u5], "Concat", [1])
    p4 = b.add(c, "C3k2", [512, False])
    p4 = caa(p4, 512, caa_kernels[1])
    outs.append(p4)

    # P5: down(P4) + P5 + raw backbone P5
    d = b.add(p4, "Conv", [1024, 3, 2])
    c = b.add([d, b10, b8], "Concat", [1])
    p5 = b.add(c, "C3k2", [1024, True])
    p5 = caa(p5, 1024, caa_kernels[2])
    outs.append(p5)

    b.add(outs, "Detect", ["nc"], 23)
    cfg = {"nc": nc, "scale": scale, "scales": {scale: [depth, width, max_ch]},
           "backbone": b.layers[:n_backbone], "head": b.layers[n_backbone:]}
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=None, width=100)
    return out_path, b.src, len(outs)


def transfer_mapped(model, ckpt_path, src_map, n_levels):
    """Copy yolo11 COCO tensors into SDF-v2 by the layer-index map, then verify by value.
    Detect: with a P2 level, SDF level l takes yolo11 level l-1 (P2 starts random);
    nc-sized class outputs never match and start random, as in every fine-tune."""
    src_model = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].float()
    src = src_model.state_dict()
    sd = model.model.state_dict()
    det_dst = len(model.model.model) - 1
    shift = n_levels - 3
    upd, per_layer = {}, {}
    for k, v in sd.items():
        parts = k.split(".")
        dst = int(parts[1])
        rest = parts[2:]
        if dst not in src_map:
            continue
        if dst == det_dst and rest[0] in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            lvl = int(rest[1]) - shift
            if lvl < 0:
                continue
            rest = [rest[0], str(lvl), *rest[2:]]
        sk = ".".join(["model", str(src_map[dst]), *rest])
        if sk in src and src[sk].shape == v.shape:
            upd[k] = src[sk]
    sd.update(upd)
    model.model.load_state_dict(sd)
    after = model.model.state_dict()
    for k, v in after.items():
        if not v.is_floating_point():
            continue
        dst = int(k.split(".")[1])
        t, e = per_layer.get(dst, (0, 0))
        per_layer[dst] = (t + 1, e + int(k in upd and torch.equal(v.cpu(), upd[k].to(v.dtype))))
    total = sum(t for t, _ in per_layer.values())
    equal = sum(e for _, e in per_layer.values())
    ht, he = per_layer.get(det_dst, (0, 0))
    print(f"SDF-v2 weight transfer: {equal}/{total} tensors from {ckpt_path.name}; "
          f"Detect head {he}/{ht}", flush=True)
    return {"tensors_total": total, "tensors_pretrained": equal, "head_pretrained": f"{he}/{ht}",
            "layers_fully_pretrained": [i for i, (t, e) in sorted(per_layer.items()) if t == e],
            "layers_random": [i for i, (t, e) in sorted(per_layer.items()) if e == 0]}
