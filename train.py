

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Dict
from collections import defaultdict, deque


from networks.my_net_2d_unet_msu import HCNet_Net_2D, DoubleConv, Encoder, UpBlock, Decoder, SideConv
from dataloaders.kvasir_dataset import KvasirSeg2D, Compose, RandomRotFlip2D, TwoStreamBatchSampler
#from dataloaders.OCTAdataset import OCTADataset, Compose, RandomRotFlip2D, TwoStreamBatchSampler
from utils import ramps, losses

from Model.sam.build_sam import build_sam_vit_b


from APGD_dual_decoder import (
    DualDecoderAPGD,
    generate_pseudo_label_with_consistency,
    extract_boundary_from_mask,
    extract_skeleton_from_mask,
)


print(torch.cuda.is_available())
def _format_params_m(num_params: int) -> float:
    return num_params / 1e6

def _format_flops_g(flops: float) -> float:
    return flops / 1e9

class _ModelProfileWrapper(nn.Module):
    """"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        try:
            out = self.model(x, en=None, gt_mask=None)
        except TypeError:
            out = self.model(x)

        if isinstance(out, (list, tuple)):
            for item in out:
                if torch.is_tensor(item):
                    return item
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        if torch.is_tensor(sub_item):
                            return sub_item
        if torch.is_tensor(out):
            return out
        raise RuntimeError('Unable to find tensor output for FLOPs profiling.')

def compute_model_complexity(model, image_size=512, in_channels=3, device='cpu'):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gflops = None
    flops_backend = None

    try:
        from thop import profile

        was_training = model.training
        model.eval()
        wrapper = _ModelProfileWrapper(model).to(device)
        dummy = torch.randn(1, in_channels, image_size, image_size, device=device)

        with torch.no_grad():
            flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)

        gflops = _format_flops_g(flops)
        flops_backend = 'thop'

        if was_training:
            model.train()
    except Exception as e:
        flops_backend = f'unavailable ({str(e)})'

    return total_params, trainable_params, gflops, flops_backend
# ============================================================================
# ============================================================================
class SAMFeatureAdapter(nn.Module):
    def __init__(self, sam_dim=768, unet_ch=256, target_size=32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(sam_dim, sam_dim // 4),
            nn.GELU(),
            nn.Linear(sam_dim // 4, unet_ch),
        )
        self.target_size = target_size
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        x = self.adapter(x)
        x = x.permute(0, 3, 1, 2)
        if self.target_size != x.shape[2]:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                            mode='bilinear', align_corners=True)
        return x * self.scale

# ============================================================================
# ============================================================================
class EncoderWithSAM(nn.Module):
    def __init__(self, n_channels=3, n_filters=16, normalization='batchnorm',
                 has_dropout=False, sam_dim=768, image_size=512):
        super().__init__()
        self.has_dropout = has_dropout
        self.image_size = image_size

        self.conv1 = DoubleConv(n_channels, n_filters, normalization)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(n_filters, n_filters * 2, normalization)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(n_filters * 2, n_filters * 4, normalization)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(n_filters * 4, n_filters * 8, normalization)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(n_filters * 8, n_filters * 16, normalization)
        self.dropout = nn.Dropout2d(p=0.3, inplace=False)

        base = image_size
        self.sam_adapters = nn.ModuleList([
            SAMFeatureAdapter(sam_dim, n_filters, base),
            SAMFeatureAdapter(sam_dim, n_filters*2, base//2),
            SAMFeatureAdapter(sam_dim, n_filters*4, base//4),
            SAMFeatureAdapter(sam_dim, n_filters*8, base//8),
            SAMFeatureAdapter(sam_dim, n_filters*16, base//16),
        ])

        self.fusion_gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid())
            for ch in [n_filters, n_filters*2, n_filters*4, n_filters*8, n_filters*16]
        ])
        self.sam_layers = [1, 3, 6, 9, 12]

    def _fuse_sam(self, unet_feat, sam_feat, level):
        if sam_feat.shape[2:] != unet_feat.shape[2:]:
            sam_feat = F.interpolate(sam_feat, size=unet_feat.shape[2:],
                                    mode='bilinear', align_corners=True)
        concat = torch.cat([unet_feat, sam_feat], dim=1)
        gate = self.fusion_gates[level](concat)
        return unet_feat + gate * sam_feat

    def forward(self, input, en=None, sam_features=None):
        if en is None:
            en = []

        sam_adapted = [None] * 5
        if sam_features is not None:
            for i, layer_idx in enumerate(self.sam_layers):
                key = f'block{layer_idx}'
                if key in sam_features:
                    sam_adapted[i] = self.sam_adapters[i](sam_features[key])

        x1 = self.conv1(input)
        if len(en) != 0: x1 = x1 + en[3]
        if sam_adapted[0] is not None: x1 = self._fuse_sam(x1, sam_adapted[0], 0)
        x1_pool = self.pool1(x1)

        x2 = self.conv2(x1_pool)
        if len(en) != 0: x2 = x2 + en[2]
        if sam_adapted[1] is not None: x2 = self._fuse_sam(x2, sam_adapted[1], 1)
        x2_pool = self.pool2(x2)

        x3 = self.conv3(x2_pool)
        if len(en) != 0: x3 = x3 + en[1]
        if sam_adapted[2] is not None: x3 = self._fuse_sam(x3, sam_adapted[2], 2)
        x3_pool = self.pool3(x3)

        x4 = self.conv4(x3_pool)
        if len(en) != 0: x4 = x4 + en[0]
        if sam_adapted[3] is not None: x4 = self._fuse_sam(x4, sam_adapted[3], 3)
        x4_pool = self.pool4(x4)

        x5 = self.conv5(x4_pool)
        # if len(en) != 0: x5 = x5 + en[0]
        # if sam_adapted[4] is not None: x5 = self._fuse_sam(x5, sam_adapted[4], 4)
        if sam_adapted[4] is not None: x5 = self._fuse_sam(x5, sam_adapted[4], 4)

        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]

# ============================================================================
# ============================================================================
class HCNet_Net_2D_WithSAM(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16,
                 normalization='batchnorm', has_dropout=True,
                 sam_model=None, sam_dim=768, image_size=512,
                 use_APGD=True):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.use_APGD = use_APGD

        self.sam_output_size = image_size // 16

        self.sam_encoder = sam_model.image_encoder
        self.sam_prompt_encoder = sam_model.prompt_encoder
        self.sam_mask_decoder = sam_model.mask_decoder

        for name, param in self.sam_encoder.named_parameters():
            if 'Adapter' not in name and 'adapter' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for name, param in self.sam_mask_decoder.named_parameters():
            if 'Adapter' in name or 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False

        self.encoder = EncoderWithSAM(n_channels, n_filters, normalization,
                                       has_dropout, sam_dim, image_size)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization,
                                has_dropout, False, up_type=0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization,
                                has_dropout, False, up_type=1)
        self.sideconv1 = SideConv(n_classes=n_classes, n_filters=n_filters)

        from networks.my_net_2d_unet_msu import MultiScaleFeedbackModule

        channels_list = [n_filters * 8, n_filters * 4, n_filters * 2, n_filters]
        self.multiscale_feedback = MultiScaleFeedbackModule(
            channels_list=channels_list,
            use_bn=True,
            use_relu=False
        )

        if use_APGD:
            self.APGD = DualDecoderAPGD(
                feature_dim=n_filters * 16,
                proj_dim=128,
                temperature=0.1,
            )
    @staticmethod
    def _mask_to_box(mask):
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        B = mask.shape[0]
        H, W = mask.shape[1], mask.shape[2]
        device = mask.device
        boxes = []
        for b in range(B):
            m = (mask[b] > 0)
            if m.sum() == 0:
                boxes.append([0.0, 0.0, float(W), float(H)])
            else:
                rows = torch.any(m, dim=1)
                cols = torch.any(m, dim=0)
                y_indices = torch.where(rows)[0]
                x_indices = torch.where(cols)[0]
                x1 = x_indices[0].float()
                x2 = x_indices[-1].float()
                y1 = y_indices[0].float()
                y2 = y_indices[-1].float()
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
        return torch.tensor(boxes, dtype=torch.float32, device=device)

    def _sam_decoder_forward(self, sam_embedding, gt_mask):
        B = sam_embedding.shape[0]
        device = sam_embedding.device
        sam_h, sam_w = sam_embedding.shape[2], sam_embedding.shape[3]

        boxes = self._mask_to_box(gt_mask)
        box_embeddings = self.sam_prompt_encoder._embed_boxes(boxes)
        sparse_embeddings = box_embeddings

        dense_embeddings = self.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B, -1,
            self.sam_prompt_encoder.image_embedding_size[0],
            self.sam_prompt_encoder.image_embedding_size[1]
        )

        image_pe = self.sam_prompt_encoder.get_dense_pe()
        if image_pe.shape[2:] != (sam_h, sam_w):
            image_pe = F.interpolate(image_pe, size=(sam_h, sam_w),
                                    mode='bilinear', align_corners=False)

        if dense_embeddings.shape[2:] != (sam_h, sam_w):
            dense_embeddings = F.interpolate(dense_embeddings, size=(sam_h, sam_w),
                                            mode='bilinear', align_corners=False)

        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=sam_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return {'masks': low_res_masks, 'iou_pred': iou_predictions}

    def forward(self, input, en=None, gt_mask=None):
        # 1. SAM Encoder
        sam_result = self.sam_encoder.forward_multiscale(input, return_intermediate=True)
        sam_embedding = sam_result['embedding']
        sam_features = sam_result['intermediate_features']

        # 2. UNet Encoder
        features = self.encoder(input, en=en, sam_features=sam_features)

        # 3. UNet Decoder
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv1(stage_feat1)

        # 4. SAM Decoder
        sam_output = None
        if self.training and gt_mask is not None:
            sam_output = self._sam_decoder_forward(sam_embedding[:gt_mask.shape[0]], gt_mask)

        x5_feat1 = stage_feat1[0]  # Decoder1的bottleneck [B, 256, h, w]
        x5_feat2 = stage_feat2[0]  # Decoder2的bottleneck [B, 256, h, w]

        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, sam_output, (x5_feat1, x5_feat2)

# ============================================================================
# ============================================================================
def compute_sam_loss(sam_output, labels, image_size=512):
    sam_masks = sam_output['masks']
    if sam_masks.shape[2] != image_size:
        sam_masks = F.interpolate(sam_masks, size=(image_size, image_size),
                                 mode='bilinear', align_corners=False)
    gt = (labels == 1).float()
    pred = sam_masks.squeeze(1)
    pred_sigmoid = torch.sigmoid(pred)
    smooth = 1e-5
    inter = (pred_sigmoid * gt).sum(dim=(1, 2))
    union = pred_sigmoid.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice_loss = 1 - (2 * inter + smooth) / (union + smooth)
    return dice_loss.mean()

def get_current_consistency_weight(iter_or_epoch, consistency=1.0, rampup=40.0):
    return consistency * ramps.sigmoid_rampup(iter_or_epoch, rampup)

def set_seed(seed=1337, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True

def dice_iou_from_logits(logits, gt, eps=1e-6):
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1).long()
        pred_fg = (pred == 1).float()
        gt_fg = (gt == 1).float()
        inter = (pred_fg * gt_fg).sum(dim=(1, 2))
        pred_sum = pred_fg.sum(dim=(1, 2))
        gt_sum = gt_fg.sum(dim=(1, 2))
        dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
        union = pred_sum + gt_sum - inter
        iou = (inter + eps) / (union + eps)
        return dice.mean().item(), iou.mean().item()

def validate(model, val_loader, device):
    model.eval()
    dices_1, ious_1 = [], []
    dices_2, ious_2 = [], []
    dices_avg, ious_avg = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs1, outputs2, _, _, _, _ = model(images, en=None, gt_mask=None)

            d1, j1 = dice_iou_from_logits(outputs1, labels)
            dices_1.append(d1); ious_1.append(j1)
            d2, j2 = dice_iou_from_logits(outputs2, labels)
            dices_2.append(d2); ious_2.append(j2)
            outputs_avg = (outputs1 + outputs2) / 2.0
            d_avg, j_avg = dice_iou_from_logits(outputs_avg, labels)
            dices_avg.append(d_avg); ious_avg.append(j_avg)

    model.train()
    return {
        "dec1": (float(np.mean(dices_1)), float(np.mean(ious_1))),
        "dec2": (float(np.mean(dices_2)), float(np.mean(ious_2))),
        "avg":  (float(np.mean(dices_avg)), float(np.mean(ious_avg))),
    }

class MorphologyConfidenceTracker:
    """
    - region(mask) history
    - boundary history
    - skeleton history
    """

    def __init__(self, history_len=6, skeleton_iters=20):
        self.history_len = history_len
        self.skeleton_iters = skeleton_iters

        self.mask_history = defaultdict(lambda: deque(maxlen=history_len))
        self.boundary_history = defaultdict(lambda: deque(maxlen=history_len))
        self.skel_history = defaultdict(lambda: deque(maxlen=history_len))

        self.epoch_cache = {}

    @torch.no_grad()
    def cache_current_epoch(self, sample_ids, pseudo_labels):
        """
        sample_ids: [B]
        pseudo_labels: [B,H,W]
        """
        masks = pseudo_labels.float().detach().cpu()
        boundaries = extract_boundary_from_mask(pseudo_labels).detach().cpu()
        skels = extract_skeleton_from_mask(pseudo_labels, iters=self.skeleton_iters).detach().cpu()

        for b in range(pseudo_labels.shape[0]):
            sid = int(sample_ids[b])
            self.epoch_cache[sid] = {
                'mask': masks[b],
                'boundary': boundaries[b],
                'skel': skels[b]
            }

    def finalize_epoch(self):
        for sid, item in self.epoch_cache.items():
            self.mask_history[sid].append(item['mask'])
            self.boundary_history[sid].append(item['boundary'])
            self.skel_history[sid].append(item['skel'])
        self.epoch_cache.clear()

    @torch.no_grad()
    def _history_conf(self, hist_dict, sample_ids, H, W, device):
        conf_list = []

        for b in range(len(sample_ids)):
            sid = int(sample_ids[b])
            hist = hist_dict[sid]

            if len(hist) < self.history_len:
                conf_list.append(torch.ones((H, W), device=device))
                continue

            hist_stack = torch.stack(list(hist), dim=0).to(device)  # [T,H,W]
            p = hist_stack.float().mean(dim=0)


            conf = 1.0 - 4.0 * p * (1.0 - p)
            conf = conf.clamp(0.0, 1.0)
            conf_list.append(conf)

        return torch.stack(conf_list, dim=0)

    def get_region_confidence(self, sample_ids, pseudo_labels):
        B, H, W = pseudo_labels.shape
        return self._history_conf(self.mask_history, sample_ids, H, W, pseudo_labels.device)

    def get_boundary_confidence(self, sample_ids, pseudo_labels):
        B, H, W = pseudo_labels.shape
        return self._history_conf(self.boundary_history, sample_ids, H, W, pseudo_labels.device)

    def get_skeleton_confidence(self, sample_ids, pseudo_labels):
        B, H, W = pseudo_labels.shape
        return self._history_conf(self.skel_history, sample_ids, H, W, pseudo_labels.device)
# ============================================================================
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/FIVES')
    parser.add_argument('--exp', type=str, default='FIVES')
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--labeled_bs', type=int, default=2)
    parser.add_argument('--labelnum', type=int, default=54)#kavsir-80  promise-3  cvc-clinicDB-48  cvc-colondb-30  ETIS-15  CHASE-1 DRIVE-2
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--base_lr', type=float, default=0.01)#0.01
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--consistency', type=float, default=2.0)
    parser.add_argument('--consistency_rampup', type=float, default=3000.0)
    parser.add_argument('--iters_feedback', type=int, default=2)#2
    parser.add_argument('--lam', type=float, default=0.001)

    # SAM参数
    parser.add_argument('--sam_checkpoint', type=str, default=r'./sam_vit_b_01ec64.pth')
    parser.add_argument('--use_sam', action='store_true', default=True)
    parser.add_argument('--sam_loss_weight', type=float, default=0.3)#0.3
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mod', type=str, default='sam_adpt')
    parser.add_argument('--thd', action='store_true', default=False)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--point_nums', type=int, default=5)
    parser.add_argument('--box_nums', type=int, default=1)

    parser.add_argument('--use_APGD', action='store_true', default=True)
    parser.add_argument('--pseudo_method', type=str, default='auto',
                        choices=['auto', 'average', 'entropy', 'intersection', 'union'],
                        help='pseudo_method')
    parser.add_argument('--APGD_weight', type=float, default=0.3,
                        help='APGD_weight')
    parser.add_argument('--APGD_threshold', type=float, default=0.9,
                        help='APGD_threshold')
    parser.add_argument('--APGD_rampup', type=float, default=3000.0,
                        help='APGD_rampup')

    parser.add_argument('--topo_history_len', type=int, default=4,
                        help='history epoch num')
    parser.add_argument('--topo_skeleton_iters', type=int, default=20,
                        help='topo_skeleton_iters')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    snapshot_path = os.path.join("./models_result", args.exp + f"_{args.labelnum}labels")
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Image size: {args.image_size}")
    logging.info(f"Using Box Prompt for SAM")
    logging.info(f"APGD: {args.use_APGD}, Pseudo Method: {args.pseudo_method}")

    set_seed(args.seed, deterministic=True)

    # Dataset
    tfm = Compose([RandomRotFlip2D()])
    train_set = KvasirSeg2D(args.data_root, split="train",
                            image_size=tuple(args.image_size), transform=tfm)

    if args.max_samples > 0:
        train_indices = list(range(min(args.max_samples, len(train_set))))
    else:
        train_indices = list(range(len(train_set)))

    labelnum = min(args.labelnum, len(train_indices))
    rng = np.random.RandomState(args.seed)
    all_ids = np.array(train_indices)
    rng.shuffle(all_ids)

    labeled_idxs = all_ids[:labelnum].tolist()
    unlabeled_idxs = all_ids[labelnum:].tolist()

    if len(unlabeled_idxs) == 0:
        raise ValueError("Unlabeled set is empty.")

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs,
        batch_size=args.batch_size,
        secondary_batch_size=args.batch_size - args.labeled_bs
    )

    train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                              num_workers=0, pin_memory=True)
    val_set = KvasirSeg2D(args.data_root, split="val",
                          image_size=tuple(args.image_size), transform=None)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)


    num_classes = 2
    img_size = args.image_size[0]

    if args.use_sam:

        args.image_size = img_size
        sam_model = build_sam_vit_b(args, checkpoint=args.sam_checkpoint)


        model = HCNet_Net_2D_WithSAM(
            n_channels=3, n_classes=num_classes, n_filters=16,
            normalization='batchnorm', has_dropout=True,
            sam_model=sam_model, sam_dim=768,
            image_size=img_size,
            use_APGD=args.use_APGD
        ).to(device)

        total_params, trainable_params, gflops, flops_backend = compute_model_complexity(
            model=model,
            image_size=img_size,
            in_channels=args.in_channels,
            device=device
        )


    else:
        model = HCNet_Net_2D(n_channels=3, n_classes=num_classes,
                             normalization='batchnorm', has_dropout=True).to(device)

    model.train()
    topo_tracker = MorphologyConfidenceTracker(
        history_len=args.topo_history_len,
        skeleton_iters=args.topo_skeleton_iters
    )
    # 训练
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    iter_num = 0
    max_epoch = args.max_iterations // len(train_loader) + 1
    best_dice = 0.0
    best_iter = 0
    iterator = tqdm(range(max_epoch), ncols=80)

    for epoch_num in iterator:
        for sampled_batch in train_loader:
            images = sampled_batch['image'].to(device)
            labels = sampled_batch['label'].to(device)

            en = []
            for t in range(args.iters_feedback):
                if t == 0:

                    outputs1, outputs2, masks, stage_out1, sam_output, bottleneck_features = model(
                        images, en=None, gt_mask=labels[:args.labeled_bs]
                    )
                else:
                    outputs1, outputs2, masks, stage_out1, sam_output, bottleneck_features = model(
                        images, en=en, gt_mask=labels[:args.labeled_bs]
                    )


                feedback_list = model.multiscale_feedback(
                    stage_feat1=masks[1][1:],
                    stage_feat2=masks[0][1:],
                    use_learned_weights=False
                )


                en = [args.lam * fb.detach() for fb in feedback_list]

                outputs_soft1 = F.softmax(outputs1, dim=1)
                outputs_soft2 = F.softmax(outputs2, dim=1)

                out5, out4, out3, out2, out1 = stage_out1
                out1_soft = F.softmax(out1, dim=1)
                out2_soft = F.softmax(out2, dim=1)
                out3_soft = F.softmax(out3, dim=1)
                out4_soft = F.softmax(out4, dim=1)
                out5_soft = F.softmax(out5, dim=1)

                labeled_bs = args.labeled_bs
                gt = (labels[:labeled_bs] == 1)

                loss_sup1 = losses.dice_loss(outputs_soft1[:labeled_bs, 1, :, :], gt)
                loss_sup2 = losses.dice_loss(outputs_soft2[:labeled_bs, 1, :, :], gt)
                loss_sup = loss_sup1 + loss_sup2

                los1 = losses.dice_loss(out1_soft[:labeled_bs, 1, :, :], gt)
                los2 = losses.dice_loss(out2_soft[:labeled_bs, 1, :, :], gt)
                los3 = losses.dice_loss(out3_soft[:labeled_bs, 1, :, :], gt)
                los4 = losses.dice_loss(out4_soft[:labeled_bs, 1, :, :], gt)
                los5 = losses.dice_loss(out5_soft[:labeled_bs, 1, :, :], gt)
                loss_ds = 0.8*los1 + 0.6*los2 + 0.4*los3 + 0.2*los4 + 0.1*los5


                cons_w = get_current_consistency_weight(iter_num, args.consistency,
                                                        args.consistency_rampup)
                loss_cons = losses.mse_loss(outputs_soft1, outputs_soft2) * cons_w


                loss_sam = torch.tensor(0.0, device=device)
                if args.use_sam and sam_output is not None:
                    loss_sam = compute_sam_loss(sam_output, labels[:labeled_bs], img_size)
                    loss_sam = loss_sam * args.sam_loss_weight


                loss_APGD = torch.tensor(0.0, device=device)
                if args.use_APGD and model.use_APGD:
                    x5_feat1, x5_feat2 = bottleneck_features

                    feat1_labeled = x5_feat1[:labeled_bs]
                    feat2_labeled = x5_feat2[:labeled_bs]
                    feat1_unlabeled = x5_feat1[labeled_bs:]
                    feat2_unlabeled = x5_feat2[labeled_bs:]

                    unlabeled_ids = sampled_batch['idx'][labeled_bs:].to(device)

                    pseudo_labels, prob_confidence, auto_info = generate_pseudo_label_with_consistency(
                        outputs1[labeled_bs:],
                        outputs2[labeled_bs:],
                        threshold=args.APGD_threshold,
                        method=args.pseudo_method
                    )

                    morph_types = auto_info['morph_types']
                    used_methods = auto_info['used_methods']

                    region_conf = topo_tracker.get_region_confidence(unlabeled_ids, pseudo_labels)
                    boundary_conf = topo_tracker.get_boundary_confidence(unlabeled_ids, pseudo_labels)
                    skeleton_conf = topo_tracker.get_skeleton_confidence(unlabeled_ids, pseudo_labels)

                    morph_conf_list = []
                    for b, morph_type in enumerate(morph_types):
                        if morph_type == 'tubular':

                            morph_conf = (
                                    0.2 * region_conf[b] +
                                    0.2 * boundary_conf[b] +
                                    0.6 * skeleton_conf[b]
                            )
                        elif morph_type == 'blob':

                            morph_conf = (
                                    0.55 * region_conf[b] +
                                    0.35 * boundary_conf[b] +
                                    0.10 * skeleton_conf[b]
                            )
                        else:

                            morph_conf = (
                                    0.7 * region_conf[b] +
                                    0.3 * boundary_conf[b]
                            )
                            morph_conf = 0.6 * morph_conf

                        morph_conf_list.append(morph_conf)

                    morph_confidence = torch.stack(morph_conf_list, dim=0)

                    confidence_mask = prob_confidence * morph_confidence

                    confidence_mask = 0.1 + 0.9 * confidence_mask

                    topo_tracker.cache_current_epoch(unlabeled_ids, pseudo_labels)

                    if len(used_methods) > 0:
                        used_method = used_methods[0]
                    else:
                        used_method = args.pseudo_method

                    if iter_num > 0 and iter_num % 1000 == 0:
                        with torch.no_grad():
                            feat_unlabeled_vis = (feat1_unlabeled + feat2_unlabeled) / 2.0
                            proj_features = model.APGD.projection(feat_unlabeled_vis)


                    # 计算对比损失
                    loss_APGD = model.APGD(
                        feat1_labeled=feat1_labeled,
                        feat2_labeled=feat2_labeled,
                        labels=labels[:labeled_bs],
                        feat1_unlabeled=feat1_unlabeled,
                        feat2_unlabeled=feat2_unlabeled,
                        pseudo_labels=pseudo_labels,
                        confidence_mask=confidence_mask
                    )


                    APGD_w = args.APGD_weight * ramps.sigmoid_rampup(iter_num, args.APGD_rampup)
                    loss_APGD = loss_APGD * APGD_w

                loss = loss_sup + 0.4*loss_ds + loss_cons + loss_sam + loss_APGD

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            iter_num += 1

            log_msg = (f"epoch {epoch_num + 1}/{max_epoch}, iter {iter_num}: "
                       f"loss={loss.item():.4f}, "
                       f"sup={loss_sup.item():.4f}, ds={loss_ds.item():.4f}, "
                       f"cons={loss_cons.item():.4f}")
            if args.use_sam:
                log_msg += f", sam={loss_sam.item():.4f}"
            if args.use_APGD:
                if args.use_APGD:
                    if args.pseudo_method == 'auto' and 'morph_types' in locals():
                        morph_summary = ",".join(morph_types[:min(2, len(morph_types))])
                        method_summary = ",".join(used_methods[:min(2, len(used_methods))])
                        log_msg += f", APGD={loss_APGD.item():.4f}, mode={method_summary}, morph={morph_summary}"
                    else:
                        log_msg += f", APGD={loss_APGD.item():.4f}, mode={used_method}"
            logging.info(log_msg)

            if iter_num % 2500 == 0:
                lr_ = args.base_lr * (0.1 ** (iter_num // 2500))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_

            if iter_num % 1000 == 0:
                val_res = validate(model, val_loader, device)
                d1, j1 = val_res["dec1"]
                d2, j2 = val_res["dec2"]
                davg, javg = val_res["avg"]

                logging.info(f"[VAL] iter {iter_num} | Dec1 Dice={d1:.4f} | "
                           f"Dec2 Dice={d2:.4f} | AVG Dice={davg:.4f}")

                if args.use_APGD and model.use_APGD:
                    proto_info = model.APGD.get_prototypes()
                    logging.info(f"[APGD] Prototypes initialized: {proto_info['initialized_1']}")


                    with torch.no_grad():
                        x5_feat1, x5_feat2 = bottleneck_features
                        proj_features = model.APGD.projection((x5_feat1 + x5_feat2) / 2)

                if davg > best_dice:
                    best_dice = davg
                    best_iter = iter_num
                    torch.save(model.state_dict(),
                              os.path.join(snapshot_path, "best_model.pth"))
                    logging.info(f"[VAL] New best Dice={best_dice:.4f}")



                torch.save(model.state_dict(),
                          os.path.join(snapshot_path, f"iter_{iter_num}.pth"))

                stage_feat1 = masks[1]  # decoder1: [x5, d4, d3, d2, d1]
                stage_feat2 = masks[0]  # decoder2: [x5, d4, d3, d2, d1]

            if iter_num >= args.max_iterations:
                iterator.close()
                break
        topo_tracker.finalize_epoch()
        if iter_num >= args.max_iterations:
            break

    logging.info(f"Best Dice={best_dice:.4f} at iter {best_iter}")


if __name__ == "__main__":
    main()
