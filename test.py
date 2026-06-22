
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.my_net_2d_unet_msu import HCNet_Net_2D, DoubleConv, Encoder, UpBlock, Decoder, SideConv
from dataloaders.kvasir_dataset import KvasirSeg2D

from Model.sam.build_sam import build_sam_vit_b


# ============================================================================
# ============================================================================
class SAMFeatureAdapter(nn.Module):

    def __init__(self, sam_dim=768, unet_ch=256, target_size=16):
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


class EncoderWithSAM(nn.Module):

    def __init__(self, n_channels=3, n_filters=16, normalization='batchnorm',
                 has_dropout=False, sam_dim=768, image_size=256):
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
            SAMFeatureAdapter(sam_dim, n_filters * 2, base // 2),
            SAMFeatureAdapter(sam_dim, n_filters * 4, base // 4),
            SAMFeatureAdapter(sam_dim, n_filters * 8, base // 8),
            SAMFeatureAdapter(sam_dim, n_filters * 16, base // 16),
        ])

        self.fusion_gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid())
            for ch in [n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16]
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
        if len(en) != 0: x1 = x1 + en[4]
        if sam_adapted[0] is not None: x1 = self._fuse_sam(x1, sam_adapted[0], 0)
        x1_pool = self.pool1(x1)

        x2 = self.conv2(x1_pool)
        if len(en) != 0: x2 = x2 + en[3]
        if sam_adapted[1] is not None: x2 = self._fuse_sam(x2, sam_adapted[1], 1)
        x2_pool = self.pool2(x2)

        x3 = self.conv3(x2_pool)
        if len(en) != 0: x3 = x3 + en[2]
        if sam_adapted[2] is not None: x3 = self._fuse_sam(x3, sam_adapted[2], 2)
        x3_pool = self.pool3(x3)

        x4 = self.conv4(x3_pool)
        if len(en) != 0: x4 = x4 + en[1]
        if sam_adapted[3] is not None: x4 = self._fuse_sam(x4, sam_adapted[3], 3)
        x4_pool = self.pool4(x4)

        x5 = self.conv5(x4_pool)
        if len(en) != 0: x5 = x5 + en[0]
        if sam_adapted[4] is not None: x5 = self._fuse_sam(x5, sam_adapted[4], 4)

        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


class HCNet_Net_2D_WithSAM(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16,
                 normalization='batchnorm', has_dropout=True,
                 sam_model=None, sam_dim=768, image_size=256):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
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

        self.encoder = EncoderWithSAM(n_channels, n_filters, normalization,
                                      has_dropout, sam_dim, image_size)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization,
                                has_dropout, False, up_type=0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization,
                                has_dropout, False, up_type=1)
        self.sideconv1 = SideConv(n_classes=n_classes, n_filters=n_filters)

        self._build_mask_downscaler(image_size)

    def _build_mask_downscaler(self, image_size):
        sam_out = image_size // 16
        layers = []
        in_ch = 1
        current_size = image_size
        channels = [4, 16, 64, 128, 256]
        idx = 0

        while current_size > sam_out and idx < len(channels):
            out_ch = channels[idx]
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2))
            layers.append(nn.GELU())
            in_ch = out_ch
            current_size = current_size // 2
            idx += 1

        if in_ch != 256:
            layers.append(nn.Conv2d(in_ch, 256, kernel_size=1))
            layers.append(nn.GELU())

        self.mask_downscaler = nn.Sequential(*layers)

    def forward(self, input, en=None, gt_mask=None, return_sam_output=False):
        # 1. SAM Encoder
        sam_result = self.sam_encoder.forward_multiscale(input, return_intermediate=True)
        sam_embedding = sam_result['embedding']
        sam_features = sam_result['intermediate_features']

        # 2. UNet Encoder (融合SAM特征)
        features = self.encoder(input, en=en, sam_features=sam_features)

        # 3. UNet Decoder
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv1(stage_feat1)

        # 4. SAM Decoder输出
        sam_output = None
        if return_sam_output:
            sam_output = self._get_sam_output(sam_embedding)

        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, sam_output

    def _get_sam_output(self, sam_embedding):

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )

        # SAM Mask Decoder
        low_res_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=sam_embedding,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        sam_output = F.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )

        return sam_output



# ============================================================================
# ============================================================================
def dice_score(pred, gt, eps=1e-6):
    """"""
    pred = pred.flatten()
    gt = gt.flatten()
    inter = (pred * gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)
def iou_score(pred, gt, eps=1e-6):
    """"""
    pred = pred.flatten()
    gt = gt.flatten()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + eps) / (union + eps)
def hausdorff_distance(pred, gt):
    try:
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial.distance import directed_hausdorff

        if pred.sum() == 0 or gt.sum() == 0:
            return float('inf')

        pred_points = np.argwhere(pred > 0)
        gt_points = np.argwhere(gt > 0)

        d1 = directed_hausdorff(pred_points, gt_points)[0]
        d2 = directed_hausdorff(gt_points, pred_points)[0]

        return max(d1, d2)
    except:
        return float('nan')

def acc_sen_spe(pred, gt, eps=1e-6):

    pred = pred.astype(np.uint8).flatten()
    gt = gt.astype(np.uint8).flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    sen = TP / (TP + FN + eps)   # Sensitivity = Recall
    spe = TN / (TN + FP + eps)   # Specificity

    return acc, sen, spe


# ============================================================================
# ============================================================================
def test(model, test_loader, device, save_dir=None, image_size=512, eval_sam=True):

    model.eval()

    # UNet Decoder1
    dices_1, ious_1 = [], []
    acc_1, sen_1, spe_1 = [], [], []
    # UNet Decoder2
    dices_2, ious_2 = [], []
    acc_2, sen_2, spe_2 = [], [], []
    # UNet Average
    dices_avg, ious_avg = [], []
    acc_list, sen_list, spe_list = [], [], []
    hausdorff_list = []

    # ★ SAM
    sam_dices, sam_ious = [], []
    sam_acc_list, sam_sen_list, sam_spe_list = [], [], []
    sam_hausdorff_list = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'overlay'), exist_ok=True)
        #
        os.makedirs(os.path.join(save_dir, 'dec1_pred'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dec1_overlay'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dec2_pred'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'dec2_overlay'), exist_ok=True)
        if eval_sam:
            os.makedirs(os.path.join(save_dir, 'sam_pred'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'sam_overlay'), exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs1, outputs2, _, _, sam_output = model(
                images, en=None, gt_mask=None, return_sam_output=eval_sam
            )

            pred1 = torch.argmax(outputs1, dim=1)
            pred2 = torch.argmax(outputs2, dim=1)
            pred_avg = torch.argmax((outputs1 + outputs2) / 2.0, dim=1)

            pred1_np = pred1.cpu().numpy()
            pred2_np = pred2.cpu().numpy()
            pred_avg_np = pred_avg.cpu().numpy()
            labels_np = labels.cpu().numpy()

            sam_pred_np = None
            if sam_output is not None:
                sam_pred = (torch.sigmoid(sam_output) > 0.5).squeeze(1)
                sam_pred_np = sam_pred.cpu().numpy()

            for b in range(images.shape[0]):
                gt = (labels_np[b] == 1).astype(np.float32)

                p1 = (pred1_np[b] == 1).astype(np.float32)
                p2 = (pred2_np[b] == 1).astype(np.float32)
                p_avg = (pred_avg_np[b] == 1).astype(np.float32)

                dices_1.append(dice_score(p1, gt))
                ious_1.append(iou_score(p1, gt))
                acc1, sen1, spe1 = acc_sen_spe(p1, gt)
                acc_1.append(acc1)
                sen_1.append(sen1)
                spe_1.append(spe1)

                dices_2.append(dice_score(p2, gt))
                ious_2.append(iou_score(p2, gt))
                acc2, sen2, spe2 = acc_sen_spe(p2, gt)
                acc_2.append(acc2)
                sen_2.append(sen2)
                spe_2.append(spe2)

                dices_avg.append(dice_score(p_avg, gt))
                ious_avg.append(iou_score(p_avg, gt))

                acc, sen, spe = acc_sen_spe(p_avg, gt)
                acc_list.append(acc)
                sen_list.append(sen)
                spe_list.append(spe)

                hd = hausdorff_distance(p_avg, gt)
                if not np.isinf(hd) and not np.isnan(hd):
                    hausdorff_list.append(hd)

                if sam_pred_np is not None:
                    p_sam = sam_pred_np[b].astype(np.float32)

                    sam_dices.append(dice_score(p_sam, gt))
                    sam_ious.append(iou_score(p_sam, gt))

                    sam_acc, sam_sen, sam_spe = acc_sen_spe(p_sam, gt)
                    sam_acc_list.append(sam_acc)
                    sam_sen_list.append(sam_sen)
                    sam_spe_list.append(sam_spe)

                    sam_hd = hausdorff_distance(p_sam, gt)
                    if not np.isinf(sam_hd) and not np.isnan(sam_hd):
                        sam_hausdorff_list.append(sam_hd)

            if save_dir:
                import cv2
                for b in range(images.shape[0]):
                    save_idx = idx * test_loader.batch_size + b
                    save_name = f"{save_idx:04d}"

                    #
                    img = images[b].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    gt_mask = (labels_np[b] == 1)

                    #
                    gt_save = (labels_np[b] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, 'gt', f"{save_name}.png"), gt_save)

                    dec1_pred_save = (pred1_np[b] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, 'dec1_pred', f"{save_name}.png"), dec1_pred_save)

                    dec2_pred_save = (pred2_np[b] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, 'dec2_pred', f"{save_name}.png"), dec2_pred_save)

                    pred_save = (pred_avg_np[b] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, 'pred', f"{save_name}.png"), pred_save)

                    if sam_pred_np is not None:
                        sam_pred_save = (sam_pred_np[b] * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, 'sam_pred', f"{save_name}.png"), sam_pred_save)

                        sam_overlay = img.copy()
                        sam_mask = (sam_pred_np[b] == 1)

                        sam_overlay[sam_mask, 2] = 255
                        sam_overlay[gt_mask, 1] = 255

                        sam_overlay = cv2.addWeighted(img, 0.5, sam_overlay, 0.5, 0)
                        cv2.imwrite(os.path.join(save_dir, 'sam_overlay', f"{save_name}.png"), sam_overlay)

    results = {
        "dec1": {
            "dice": float(np.mean(dices_1)),
            "dice_std": float(np.std(dices_1)),
            "iou": float(np.mean(ious_1)),
            "iou_std": float(np.std(ious_1)),
            "acc": float(np.mean(acc_1)),
            "sen": float(np.mean(sen_1)),
            "spe": float(np.mean(spe_1)),
        },
        "dec2": {
            "dice": float(np.mean(dices_2)),
            "dice_std": float(np.std(dices_2)),
            "iou": float(np.mean(ious_2)),
            "iou_std": float(np.std(ious_2)),
            "acc": float(np.mean(acc_2)),
            "sen": float(np.mean(sen_2)),
            "spe": float(np.mean(spe_2)),
        },
        "avg": {
            "dice": float(np.mean(dices_avg)),
            "dice_std": float(np.std(dices_avg)),
            "iou": float(np.mean(ious_avg)),
            "iou_std": float(np.std(ious_avg)),
            "acc": float(np.mean(acc_list)),
            "sen": float(np.mean(sen_list)),
            "spe": float(np.mean(spe_list)),
        },
        "hausdorff": {
            "mean": float(np.mean(hausdorff_list)) if hausdorff_list else float('nan'),
            "std": float(np.std(hausdorff_list)) if hausdorff_list else float('nan'),
        },
        "num_samples": len(dices_1),
    }

    if eval_sam and sam_dices:
        results["sam"] = {
            "dice": float(np.mean(sam_dices)),
            "dice_std": float(np.std(sam_dices)),
            "iou": float(np.mean(sam_ious)),
            "iou_std": float(np.std(sam_ious)),
            "acc": float(np.mean(sam_acc_list)),
            "sen": float(np.mean(sam_sen_list)),
            "spe": float(np.mean(sam_spe_list)),
            "hausdorff_mean": float(np.mean(sam_hausdorff_list)) if sam_hausdorff_list else float('nan'),
            "hausdorff_std": float(np.std(sam_hausdorff_list)) if sam_hausdorff_list else float('nan'),
        }

    return results


# ============================================================================
# ============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default=r'./data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--split', type=str, default='test', help='test or val')

    parser.add_argument('--model_path', type=str, default=r'./model_path', help='model_path')
    parser.add_argument('--use_sam', action='store_true', default=True)
    parser.add_argument('--sam_checkpoint', type=str, default=r'./sam_vit_b_01ec64.pth')

    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mod', type=str, default='sam_adpt')#sam_frozen ,sam_adpt
    parser.add_argument('--thd', action='store_true', default=False)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--point_nums', type=int, default=5)
    parser.add_argument('--box_nums', type=int, default=1)

    parser.add_argument('--eval_sam', action='store_true', default=False,
                        help='eval_sam')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_pred', default=True, help='save_pred')
    parser.add_argument('--save_dir', type=str, default=r'./result', help='save_dir')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    img_size = args.image_size[0]
    print(f"Image size: {img_size}x{img_size}")

    test_set = KvasirSeg2D(
        args.data_root,
        split=args.split,
        image_size=tuple(args.image_size),
        transform=None
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)
    print(f"Test samples: {len(test_set)}")

    num_classes = 2

    if args.use_sam:
        args.image_size = img_size
        sam_model = build_sam_vit_b(args, checkpoint=args.sam_checkpoint)

        model = HCNet_Net_2D_WithSAM(
            n_channels=3, n_classes=num_classes, n_filters=16,
            normalization='batchnorm', has_dropout=False,
            sam_model=sam_model, sam_dim=768,
            image_size=img_size
        ).to(device)
    else:
        model = HCNet_Net_2D(
            n_channels=3, n_classes=num_classes,
            normalization='batchnorm', has_dropout=False
        ).to(device)

    if os.path.exists(args.model_path):
        print(f"Loading model from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)

        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    save_dir = args.save_dir if args.save_pred else None
    results = test(model, test_loader, device, save_dir, img_size, eval_sam=args.eval_sam)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Image size: {img_size}x{img_size}")
    print("-" * 70)

    print("\n【UNet】")
    print(f"Decoder1:  Dice = {results['dec1']['dice']:.4f} ± {results['dec1']['dice_std']:.4f}  |  "
          f"IoU = {results['dec1']['iou']:.4f} ± {results['dec1']['iou_std']:.4f}")
    print(f"           Acc  = {results['dec1']['acc']:.4f}  |  "
          f"Sen = {results['dec1']['sen']:.4f}  |  "
          f"Spe = {results['dec1']['spe']:.4f}")

    print(f"Decoder2:  Dice = {results['dec2']['dice']:.4f} ± {results['dec2']['dice_std']:.4f}  |  "
          f"IoU = {results['dec2']['iou']:.4f} ± {results['dec2']['iou_std']:.4f}")
    print(f"           Acc  = {results['dec2']['acc']:.4f}  |  "
          f"Sen = {results['dec2']['sen']:.4f}  |  "
          f"Spe = {results['dec2']['spe']:.4f}")

    print(f"Average:   Dice = {results['avg']['dice']:.4f} ± {results['avg']['dice_std']:.4f}  |  "
          f"IoU = {results['avg']['iou']:.4f} ± {results['avg']['iou_std']:.4f}")

    print(f"           Acc  = {results['avg']['acc']:.4f}  |  "
          f"Sen = {results['avg']['sen']:.4f}  |  "
          f"Spe = {results['avg']['spe']:.4f}")

    print(f"Hausdorff: {results['hausdorff']['mean']:.2f} ± {results['hausdorff']['std']:.2f}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()