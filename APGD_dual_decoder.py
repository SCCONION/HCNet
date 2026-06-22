
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img, iters=20):
    """
    img: [B,1,H,W], float in [0,1]
    return: [B,1,H,W], binary-like skeleton
    """
    img = img.float()
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = torch.max(skel, delta)
    return (skel > 0.5).float()


def extract_skeleton_from_mask(mask, iters=20):
    """
    mask: [B,H,W] or [B,1,H,W]
    return: [B,H,W]
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1).float()
    skel = soft_skeletonize(mask, iters=iters)
    return skel.squeeze(1)


def extract_boundary_from_mask(mask):
    """
    mask: [B,H,W] or [B,1,H,W]
    return: [B,H,W]
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1).float()
    dilated = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded > 0).float()
    return boundary.squeeze(1)


@torch.no_grad()
def classify_morphology_from_prob(prob_avg,
                                  area_min=100,
                                  boundary_area_thresh=0.28,
                                  skeleton_area_thresh=0.22):
    """
    prob_avg: [B,2,H,W]
    return:
        morph_types: list[str], each in {'tubular', 'blob', 'noisy'}
        morph_stats: list[dict]
    """
    fg_mask = (prob_avg[:, 1] > 0.5).float()   # [B,H,W]
    B = fg_mask.shape[0]

    morph_types = []
    morph_stats = []

    for b in range(B):
        mask = fg_mask[b:b+1]  # [1,H,W]
        area = mask.sum().item()

        if area < area_min:
            morph_types.append('noisy')
            morph_stats.append({
                'area': area,
                'boundary_area_ratio': 0.0,
                'skeleton_area_ratio': 0.0,
            })
            continue

        boundary = extract_boundary_from_mask(mask)   # [1,H,W]
        skeleton = extract_skeleton_from_mask(mask)   # [1,H,W]

        boundary_area = boundary.sum().item()
        skeleton_area = skeleton.sum().item()

        ba_ratio = boundary_area / (area + 1e-6)
        sa_ratio = skeleton_area / (area + 1e-6)

        if ba_ratio > boundary_area_thresh and sa_ratio > skeleton_area_thresh:
            morph_type = 'tubular'
        else:
            morph_type = 'blob'

        morph_types.append(morph_type)
        morph_stats.append({
            'area': area,
            'boundary_area_ratio': ba_ratio,
            'skeleton_area_ratio': sa_ratio,
        })

    return morph_types, morph_stats

class ProjectionHead(nn.Module):
    """"""
    def __init__(self, in_dim, proj_dim=128, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, proj_dim, 1),
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = F.normalize(x, dim=1, p=2)
        return x


class DualDecoderAPGD(nn.Module):
    """
    """
    def __init__(self, feature_dim=256, proj_dim=128, temperature=0.1, 
                 ema_momentum=0.99, min_pixels=10, mode='average'):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.min_pixels = min_pixels
        self.mode = mode  # 保留兼容
        
        # 投影头
        self.projection = ProjectionHead(feature_dim, proj_dim)
        
        # 原型
        self.register_buffer('prototype_fg_1', torch.zeros(proj_dim))
        self.register_buffer('prototype_bg_1', torch.zeros(proj_dim))
        self.register_buffer('prototype_initialized_1', torch.tensor(False))
    
    def _compute_prototypes(self, features, labels):
        """"""
        B, C, h, w = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1).float()
        

        fg_mask = labels_flat > 0.5
        if fg_mask.sum() >= self.min_pixels:
            proto_fg = features_flat[fg_mask].mean(dim=0)
        else:
            proto_fg = torch.zeros(C, device=features.device)
        

        bg_mask = labels_flat < 0.5
        if bg_mask.sum() >= self.min_pixels:
            proto_bg = features_flat[bg_mask].mean(dim=0)
        else:
            proto_bg = torch.zeros(C, device=features.device)

        proto_fg = F.normalize(proto_fg, dim=0, p=2)
        proto_bg = F.normalize(proto_bg, dim=0, p=2)
        
        return proto_fg, proto_bg
    
    def _update_prototypes_ema(self, proto_fg, proto_bg):
        """"""
        if not self.prototype_initialized_1:
            self.prototype_fg_1.copy_(proto_fg)
            self.prototype_bg_1.copy_(proto_bg)
            self.prototype_initialized_1.fill_(True)
        else:
            self.prototype_fg_1.copy_(
                self.ema_momentum * self.prototype_fg_1 + 
                (1 - self.ema_momentum) * proto_fg
            )
            self.prototype_bg_1.copy_(
                self.ema_momentum * self.prototype_bg_1 + 
                (1 - self.ema_momentum) * proto_bg
            )
            self.prototype_fg_1.copy_(F.normalize(self.prototype_fg_1, dim=0, p=2))
            self.prototype_bg_1.copy_(F.normalize(self.prototype_bg_1, dim=0, p=2))
    
    def _contrastive_loss(self, features, pseudo_labels, confidence_mask=None):
        """"""
        B, C, h, w = features.shape
        device = features.device
        
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        pseudo_flat = pseudo_labels.reshape(-1).float()
        
        if confidence_mask is not None:
            conf_flat = confidence_mask.reshape(-1)
        else:
            conf_flat = torch.ones_like(pseudo_flat)
        
        sim_to_fg = torch.sum(features_flat * self.prototype_fg_1.unsqueeze(0), dim=1) / self.temperature
        sim_to_bg = torch.sum(features_flat * self.prototype_bg_1.unsqueeze(0), dim=1) / self.temperature

        fg_mask = pseudo_flat > 0.5
        if fg_mask.sum() > 0:
            logits_fg = torch.stack([sim_to_fg[fg_mask], sim_to_bg[fg_mask]], dim=1)
            loss_fg = -sim_to_fg[fg_mask] + torch.logsumexp(logits_fg, dim=1)
            loss_fg = (loss_fg * conf_flat[fg_mask]).sum() / (conf_flat[fg_mask].sum() + 1e-6)
        else:
            loss_fg = torch.tensor(0.0, device=device)

        bg_mask = pseudo_flat < 0.5
        if bg_mask.sum() > 0:
            logits_bg = torch.stack([sim_to_bg[bg_mask], sim_to_fg[bg_mask]], dim=1)
            loss_bg = -sim_to_bg[bg_mask] + torch.logsumexp(logits_bg, dim=1)
            loss_bg = (loss_bg * conf_flat[bg_mask]).sum() / (conf_flat[bg_mask].sum() + 1e-6)
        else:
            loss_bg = torch.tensor(0.0, device=device)
        
        return (loss_fg + loss_bg) / 2
    
    def forward(self, feat1_labeled, feat2_labeled, labels,
                feat1_unlabeled=None, feat2_unlabeled=None,
                pseudo_labels=None, confidence_mask=None):
        """
        """
        device = feat1_labeled.device
        h, w = feat1_labeled.shape[2], feat1_labeled.shape[3]
        
        #
        labels_down = F.interpolate(
            labels.unsqueeze(1).float(), size=(h, w), mode='nearest'
        ).squeeze(1).long()
        
        #
        feat_labeled = (feat1_labeled + feat2_labeled) / 2
        proj_labeled = self.projection(feat_labeled)
        
        #
        proto_fg, proto_bg = self._compute_prototypes(proj_labeled, labels_down)
        if self.training:
            self._update_prototypes_ema(proto_fg.detach(), proto_bg.detach())
        
        if feat1_unlabeled is None or pseudo_labels is None:
            return torch.tensor(0.0, device=device)
        
        #
        feat_unlabeled = (feat1_unlabeled + feat2_unlabeled) / 2
        proj_unlabeled = self.projection(feat_unlabeled)
        
        #
        pseudo_down = F.interpolate(
            pseudo_labels.unsqueeze(1).float(), size=(h, w), mode='nearest'
        ).squeeze(1)
        
        conf_down = None
        if confidence_mask is not None:
            conf_down = F.interpolate(
                confidence_mask.unsqueeze(1).float(), size=(h, w), mode='nearest'
            ).squeeze(1)
        
        loss = self._contrastive_loss(proj_unlabeled, pseudo_down, conf_down)
        return loss
    
    def get_prototypes(self):
        """"""
        return {
            'prototype_fg_1': self.prototype_fg_1.clone(),
            'prototype_bg_1': self.prototype_bg_1.clone(),
            'initialized_1': self.prototype_initialized_1.item()
        }


# ============================================================================
# ============================================================================
def _determine_morphology_strategy(outputs1, outputs2, threshold_ratio=0.25):
    with torch.no_grad():

        prob1 = F.softmax(outputs1, dim=1)
        prob2 = F.softmax(outputs2, dim=1)
        prob_avg = (prob1 + prob2) / 2.0
        fg_mask = (prob_avg[:, 1, :, :] > 0.5).float().unsqueeze(1)
        area = fg_mask.sum()
        if area < 100:
            return 'average'

        dilated = F.max_pool2d(fg_mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-fg_mask, kernel_size=3, stride=1, padding=1)
        boundary = dilated - eroded

        boundary_area = boundary.sum()
        ratio = (boundary_area / area).item()
        if ratio > threshold_ratio:
            return 'average'
        else:
            return 'union'
def generate_pseudo_label_with_consistency(outputs1, outputs2, threshold=0.9, method='average'):
    """
    Returns:
        pseudo_labels: [B,H,W]
        confidence_mask: [B,H,W], soft confidence in [0,1]
        auto_info: dict
    """
    prob1 = F.softmax(outputs1, dim=1)
    prob2 = F.softmax(outputs2, dim=1)
    prob_avg = (prob1 + prob2) / 2.0

    morph_types, morph_stats = classify_morphology_from_prob(prob_avg)

    if method == 'auto':
        pseudo_list = []
        conf_list = []
        used_methods = []

        for b, morph_type in enumerate(morph_types):
            if morph_type == 'tubular':
                used_method = 'average'
            elif morph_type == 'blob':
                used_method = 'union'
            else:  # noisy
                used_method = 'average'

            if used_method == 'average':
                pseudo_b, conf_b = _pseudo_average(outputs1[b:b+1], outputs2[b:b+1], threshold)
            elif used_method == 'entropy':
                pseudo_b, conf_b = _pseudo_entropy(outputs1[b:b+1], outputs2[b:b+1], threshold)
            elif used_method == 'intersection':
                pseudo_b, conf_b = _pseudo_intersection(outputs1[b:b+1], outputs2[b:b+1], threshold)
            elif used_method == 'union':
                pseudo_b, conf_b = _pseudo_union(outputs1[b:b+1], outputs2[b:b+1], threshold)
            else:
                raise ValueError(f"Unknown method: {used_method}")

            pseudo_list.append(pseudo_b)
            conf_list.append(conf_b)
            used_methods.append(used_method)

        pseudo_labels = torch.cat(pseudo_list, dim=0)
        confidence_mask = torch.cat(conf_list, dim=0)

        auto_info = {
            'mode': 'auto',
            'morph_types': morph_types,
            'morph_stats': morph_stats,
            'used_methods': used_methods
        }
        return pseudo_labels, confidence_mask, auto_info

    if method == 'average':
        pseudo_labels, confidence_mask = _pseudo_average(outputs1, outputs2, threshold)
    elif method == 'entropy':
        pseudo_labels, confidence_mask = _pseudo_entropy(outputs1, outputs2, threshold)
    elif method == 'intersection':
        pseudo_labels, confidence_mask = _pseudo_intersection(outputs1, outputs2, threshold)
    elif method == 'union':
        pseudo_labels, confidence_mask = _pseudo_union(outputs1, outputs2, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    auto_info = {
        'mode': method,
        'morph_types': morph_types,
        'morph_stats': morph_stats,
        'used_methods': [method] * outputs1.shape[0]
    }
    return pseudo_labels, confidence_mask, auto_info

def _pseudo_average(outputs1, outputs2, threshold=0.9):

    prob1 = F.softmax(outputs1, dim=1)
    prob2 = F.softmax(outputs2, dim=1)

    prob_avg = (prob1 + prob2) / 2
    max_prob, pseudo_labels = prob_avg.max(dim=1)

    pred1 = prob1.argmax(dim=1)
    pred2 = prob2.argmax(dim=1)
    consistency_mask = (pred1 == pred2).float()

    base_conf = ((max_prob - threshold) / (1.0 - threshold + 1e-6)).clamp(0, 1)
    confidence_mask = base_conf * consistency_mask

    return pseudo_labels, confidence_mask


def _pseudo_entropy(outputs1, outputs2, threshold=0.9, entropy_thresh=0.4):
    prob1 = F.softmax(outputs1, dim=1)
    prob2 = F.softmax(outputs2, dim=1)

    entropy1 = -torch.sum(prob1 * torch.log(prob1 + 1e-6), dim=1)
    entropy2 = -torch.sum(prob2 * torch.log(prob2 + 1e-6), dim=1)

    max_prob1, pred1 = prob1.max(dim=1)
    max_prob2, pred2 = prob2.max(dim=1)

    consistency_mask = (pred1 == pred2).float()

    low_entropy1 = (1.0 - entropy1 / (math.log(prob1.shape[1]) + 1e-6)).clamp(0, 1)
    low_entropy2 = (1.0 - entropy2 / (math.log(prob2.shape[1]) + 1e-6)).clamp(0, 1)
    entropy_conf = 0.5 * (low_entropy1 + low_entropy2)

    prob_avg = (prob1 + prob2) / 2
    _, pseudo_labels = prob_avg.max(dim=1)

    avg_max_prob = (max_prob1 + max_prob2) / 2.0
    prob_conf = ((avg_max_prob - threshold) / (1.0 - threshold + 1e-6)).clamp(0, 1)

    confidence_mask = consistency_mask * entropy_conf * prob_conf

    return pseudo_labels, confidence_mask


def _pseudo_intersection(outputs1, outputs2, threshold=0.9):
    prob1 = F.softmax(outputs1, dim=1)
    prob2 = F.softmax(outputs2, dim=1)

    fg_prob1 = prob1[:, 1, :, :]
    fg_prob2 = prob2[:, 1, :, :]

    pred1_fg = (fg_prob1 > 0.5).float()
    pred2_fg = (fg_prob2 > 0.5).float()

    pseudo_fg = pred1_fg * pred2_fg
    pseudo_labels = pseudo_fg.long()

    confidence = torch.min(fg_prob1, fg_prob2) * pseudo_fg + \
                 torch.min(1 - fg_prob1, 1 - fg_prob2) * (1 - pseudo_fg)

    confidence_mask = ((confidence - threshold) / (1.0 - threshold + 1e-6)).clamp(0, 1)

    return pseudo_labels, confidence_mask


def _pseudo_union(outputs1, outputs2, threshold=0.9):
    prob1 = F.softmax(outputs1, dim=1)
    prob2 = F.softmax(outputs2, dim=1)

    fg_prob1 = prob1[:, 1, :, :]
    fg_prob2 = prob2[:, 1, :, :]

    pred1_fg = (fg_prob1 > 0.5).float()
    pred2_fg = (fg_prob2 > 0.5).float()

    pseudo_fg = torch.clamp(pred1_fg + pred2_fg, 0, 1)
    pseudo_labels = pseudo_fg.long()

    confidence = torch.max(fg_prob1, fg_prob2) * pseudo_fg + \
                 torch.max(1 - fg_prob1, 1 - fg_prob2) * (1 - pseudo_fg)

    consistency = (pred1_fg == pred2_fg).float()
    confidence = confidence * (0.9 + 0.1 * consistency)

    confidence_mask = ((confidence - threshold) / (1.0 - threshold + 1e-6)).clamp(0, 1)

    return pseudo_labels, confidence_mask
