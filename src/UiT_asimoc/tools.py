# ---- This is <tools.py> ----

"""
tools module for ice-water classification with UNET
"""

import numpy as np
import math
import six
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def fill_nan_with_nearest(array):

    mask = np.where(~np.isnan(array))
    interp = NearestNDInterpolator(np.transpose(mask), array[mask])
    filled_array = interp(*np.indices(array.shape))

    return filled_array

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
    
def create_datacube(HH, HV, IA, patch_height, patch_width):
    ''' Create datacube with HH, HV, IA channels of shape (3, H*patch_height, W*patch_width)'''
    # ------------------------------------------------------------------
    # get dimensions of original image bands
    original_image_height =  IA.shape[0]
    original_image_width = IA.shape[1]
    
    # how many times does a patch fit in height and width of the original image?
    H_original = original_image_height/patch_height
    H = math.ceil(H_original)
    
    W_original = original_image_width/patch_width
    W = math.ceil(W_original)

    # initialize datacube with value -100 dB of shape nr_featuresxMxN   
    # note: image borders are padded (with -100 dB) to yield an integer nr of patches along height and width dimension
    image_datacube = np.ones((3, H * patch_height, W * patch_width))* -100
    
    # ------------------------------------------------------------------
    # fill up data cube

    # 1st dimension = noise-corrected sigma_0 HH in dB
    image_datacube[0, 0:HH.shape[0], 0:HH.shape[1]] = HH
    # clip to lower boundary value for HH
    image_datacube[0,:,:][image_datacube[0,:,:] < -35] = -35
    
    # 2nd dimension = noise-corrected sigma_0 HV in dB
    image_datacube[1, 0:HV.shape[0], 0:HV.shape[1]] = HV
    # clip to lower boundary value for HH
    image_datacube[1,:,:][image_datacube[1,:,:] < -40] = -40
    
    # 3rd dimension = IA
    image_datacube[2, 0:IA.shape[0], 0:IA.shape[1]]  = IA
    
    return image_datacube, original_image_height, original_image_width

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # ensure deterministic convolution operations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class UNet_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32, dropout_prob = 0.0)
        self.down1 = Down(32, 64, dropout_prob = 0.0)
        self.down2 = Down(64, 128, dropout_prob = 0.0)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor, dropout_prob = 0.0)

        self.bottleneck_dropout = nn.Dropout(p=0.0)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)  # torch.Size([32, 32, 224, 224])
        x2 = self.down1(x1) # [112,112]
        # print(x2.shape) # torch.Size([32, 64, 112, 112])
        x3 = self.down2(x2) # [56,56]
        # print(x3.shape)
        x4 = self.down3(x3) # [28,28]
        # print(x4.shape)
        x4 = self.bottleneck_dropout(x4)
        x = self.up1(x4, x3)

        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# the left part with horizontal convolution
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),  # Add dropout here
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)   # Add dropout here
        )

    def forward(self, x):
        return self.double_conv(x)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout_prob=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_prob=dropout_prob)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    pred_labels = iter(pred_labels)

    gt_labels = iter(gt_labels)

    n_class = 2
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.  
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class ** 2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.


    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    # add a small number to avoid NAN
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        if target.max() >= self.num_class or target.min() < 0:
            print(f"Invalid target values detected: {target}")
            raise ValueError(f"Target values must be between 0 and {self.num_class - 1}")

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def Uncertainty_overlapping_patches(model, data_loader, image_size, patch_size, n_classes=2):
    """
    Evaluates uncertainty and class predictions over an image using overlapping patches processed in batches,
    using the same Probability Calculation formula and uncertainty measure as Uncertainty_overlapping_patches.

    Args:
    - model: The neural network model for predictions.
    - data_loader: DataLoader providing batches of image patches and their coordinates.
    - image_size: Tuple of (height, width) for the full image size.
    - patch_size: The size of each square patch.
    - n_classes: The number of classes for classification tasks. Assumes binary classification.

    Returns:
    - class_mat_full: Classification results per pixel, averaged over overlaps.
    - probs_mat_full: Probability of the most probable class per pixel, averaged over overlaps.
    - uncertainty_overlap: Uncertainty map based on the variance of predictions across overlapping patches.
    """
    device = next(model.parameters()).device
    model.eval()

    # Initialize matrices to store the sum of predictions, and the count of patches covering each pixel
    class_sums = np.zeros(image_size, dtype=np.float32)
    probs_sums = np.zeros(image_size, dtype=np.float32)
    overlaps = np.zeros(image_size, dtype=np.int32)

    with torch.no_grad():
        for patches, _, coords in tqdm(data_loader, desc="Processing patches"):
            patches = patches.float().to(device)
            outputs = model(patches)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, dim=1)
            probs_max = probs.max(dim=1)[0]  # Extract the maximum probability

            for i in range(len(coords[0])):
                x, y = coords[0][i].item(), coords[1][i].item()
                x, y = int(x), int(y)
                y_end = min(y + patch_size, image_size[0])
                x_end = min(x + patch_size, image_size[1])
                valid_patch_height = y_end - y
                valid_patch_width = x_end - x

                C_curr = predicted[i].cpu().numpy()[:valid_patch_height, :valid_patch_width]
                P_curr = probs_max[i].cpu().numpy()[:valid_patch_height, :valid_patch_width]
                probs_calc = (1 - C_curr) * (1 - P_curr) + C_curr * P_curr  

                class_sums[y:y_end, x:x_end] += C_curr
                probs_sums[y:y_end, x:x_end] += probs_calc
                overlaps[y:y_end, x:x_end] += 1

    # Normalize by the number of overlaps
    class_mat_full = np.divide(class_sums, overlaps, where=overlaps > 0, out=np.zeros_like(class_sums))
    probs_mat_full = np.divide(probs_sums, overlaps, where=overlaps > 0, out=np.zeros_like(probs_sums))

    # Calculate uncertainty using overlapping patches
    uncertainty_overlap = np.zeros_like(class_mat_full)
    uncertainty_overlap[class_mat_full > 0.5] = 1 - class_mat_full[class_mat_full > 0.5]
    uncertainty_overlap[class_mat_full <= 0.5] = class_mat_full[class_mat_full <= 0.5]
    uncertainty_overlap = uncertainty_overlap * n_classes

    # probability no ice 
    probs_no_ice = 1 - probs_mat_full

    epsilon = np.finfo(float).eps

    # Calculate entropy per pixel
    shannon_entropy = -(probs_mat_full * np.log2(probs_mat_full + epsilon) +
                        probs_no_ice * np.log2(probs_no_ice + epsilon))

    return class_mat_full, probs_mat_full, overlaps, uncertainty_overlap, shannon_entropy

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <tools.py> ----

