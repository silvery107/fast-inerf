"""
Implements the PoseCNN network architecture in PyTorch.
"""
import os
import time
import random
import pyquaternion
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision
from torchvision.ops import RoIPool, box_iou
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.PROPSPoseDataset import PROPSPoseDataset
from utils.posecnn_utils import quaternion_to_matrix

_HOUGHVOTING_NUM_INLIER = 100
_HOUGHVOTING_DIRECTION_INLIER = 0.9
_LABEL2MASK_THRESHOL = 100

class FeatureExtraction(nn.Module):
    """
    Feature Embedding Module for PoseCNN. Using pretrained VGG16 network as backbone.
    """    
    def __init__(self, pretrained_model):
        super(FeatureExtraction, self).__init__()
        embedding_layers = list(pretrained_model.features)[:30]
        ## Embedding Module from begining till the first output feature map
        self.embedding1 = nn.Sequential(*embedding_layers[:23])
        ## Embedding Module from the first output feature map till the second output feature map
        self.embedding2 = nn.Sequential(*embedding_layers[23:])

        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.embedding1[i].weight.requires_grad = False
            self.embedding1[i].bias.requires_grad = False
    
    def forward(self, datadict):
        """
        feature1: [bs, 512, H/8, W/8]
        feature2: [bs, 512, H/16, W/16]
        """ 
        feature1 = self.embedding1(datadict['rgb'])
        feature2 = self.embedding2(feature1)
        return feature1, feature2

class SegmentationBranch(nn.Module):
    """
    Instance Segmentation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        ######################################################################
        # Initialize instance segmentation branch layers for PoseCNN.  #
        #                                                                    #
        # 1) Both feature1 and feature2 should be passed through a 1x1 conv  #
        # + ReLU layer (seperate layer for each feature).                    #
        #                                                                    #
        # 2) Next, intermediate features from feature1 should be upsampled   #
        # to match spatial resolution of features2.                          #
        #                                                                    #
        # 3) Intermediate features should be added, element-wise.            #
        #                                                                    #
        # 4) Final probability map generated by 1x1 conv+ReLU -> softmax     #
        #                                                                    #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        #                                                                    #
        # Note: num_classes passed as input does not include the background  #
        # our desired probability map should be over classses and background #
        # Input channels will be 512, hidden_layer_dim gives channels for    #
        # each embedding layer in this network.                              #
        ######################################################################
        self.num_classes = num_classes
        input_channels = 512
        self.conv_f1 = nn.Sequential(nn.Conv2d(input_channels, hidden_layer_dim, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True))
        self.conv_f2 = nn.Sequential(nn.Conv2d(input_channels, hidden_layer_dim, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2., mode="bilinear"))
        self.seg_prob = nn.Sequential(nn.Upsample(scale_factor=8., mode="bilinear"),
                                      nn.Conv2d(hidden_layer_dim, num_classes+1, kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Softmax2d())

        # init model params
        kaiming_normal_(self.conv_f1[0].weight.data)
        self.conv_f1[0].bias.data.zero_()
        kaiming_normal_(self.conv_f2[0].weight.data)
        self.conv_f2[0].bias.data.zero_()
        kaiming_normal_(self.seg_prob[1].weight.data)
        self.seg_prob[1].bias.data.zero_()
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            probability: Segmentation map of probability for each class at each pixel.
                probability size: (B,num_classes+1,H,W)
            segmentation: Segmentation map of class id's with highest prob at each pixel.
                segmentation size: (B,H,W)
            bbx: Bounding boxs detected from the segmentation. Can be extracted 
                from the predicted segmentation map using self.label2bbx(segmentation).
                bbx size: (N,6) with (batch_ids, x1, y1, x2, y2, cls)
        """
        probability = None
        segmentation = None
        bbx = None
        
        ######################################################################
        # Implement forward pass of instance segmentation branch.      #
        ######################################################################
        
        inter_feat = self.conv_f1(feature1) + self.conv_f2(feature2)
        probability = self.seg_prob(inter_feat)
        _, segmentation = torch.max(probability, dim=1)
        bbx = self.label2bbx(segmentation)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return probability, segmentation, bbx
    
    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)
        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx
        
        
class TranslationBranch(nn.Module):
    """
    3D Translation Estimation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()
        
        ######################################################################
        # Initialize layers of translation branch for PoseCNN.         #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        self.num_classes = num_classes
        input_channels = 512
        self.conv_f1 = nn.Sequential(nn.Conv2d(input_channels, hidden_layer_dim, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True)
                                     )
        self.conv_f2 = nn.Sequential(nn.Conv2d(input_channels, hidden_layer_dim, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample(scale_factor=2., mode="bilinear"))
        self.trans_pred = nn.Sequential(nn.Upsample(scale_factor=8., mode="bilinear"),
                                        nn.Conv2d(hidden_layer_dim, num_classes*3, kernel_size=1, stride=1, padding=0))
        
        kaiming_normal_(self.conv_f1[0].weight.data)
        self.conv_f1[0].bias.data.zero_()
        kaiming_normal_(self.conv_f2[0].weight.data)
        self.conv_f2[0].bias.data.zero_()
        kaiming_normal_(self.trans_pred[1].weight.data)
        self.trans_pred[1].bias.data.zero_()
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            translation: Map of object centroid predictions.
                translation size: (N,3*num_classes,H,W)
        """
        translation = None
        ######################################################################
        # Implement forward pass of translation branch.                #
        ######################################################################
        inter_feat = self.conv_f1(feature1) + self.conv_f2(feature2)
        translation = self.trans_pred(inter_feat)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        return translation

class RotationBranch(nn.Module):
    """
    3D Rotation Regression Module for PoseCNN. 
    """    
    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()

        ######################################################################
        # Initialize layers of rotation branch for PoseCNN.            #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        self.num_classes = num_classes
        self.roi_pool_f1 = RoIPool(roi_shape, spatial_scale=1./8.)
        self.roi_pool_f2 = RoIPool(roi_shape, spatial_scale=1./16.)
        self.roi_fc = nn.Sequential(nn.Linear(feature_dim*roi_shape*roi_shape, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 4*num_classes))
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2, bbx):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
            bbx: Bounding boxes of regions of interst (N, 5) with (batch_ids, x1, y1, x2, y2)
        Returns:
            quaternion: Regressed components of a quaternion for each class at each ROI.
                quaternion size: (N, 4*num_classes)
        """
        quaternion = None

        ######################################################################
        # Implement forward pass of rotation branch.                   #
        ######################################################################
        rois = self.roi_pool_f1(feature1, bbx) + self.roi_pool_f2(feature2, bbx)
        N = bbx.shape[0]
        # print(rois.shape)
        quaternion = self.roi_fc(rois.flatten(start_dim=1))
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return quaternion

class PoseCNN(nn.Module):
    """
    PoseCNN
    """
    def __init__(self, pretrained_backbone, models_pcd, cam_intrinsic, num_classes=10):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic

        ######################################################################
        # Initialize layers and components of PoseCNN.                 #
        #                                                                    #
        # Create an instance of FeatureExtraction, SegmentationBranch,       #
        # TranslationBranch, and RotationBranch for use in PoseCNN           #
        ######################################################################
        self.feat_extract = FeatureExtraction(pretrained_backbone)
        self.seg_branch = SegmentationBranch(num_classes=num_classes)
        self.trans_branch = TranslationBranch(num_classes=num_classes)
        self.rot_branch = RotationBranch(num_classes=num_classes)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, input_dict):
        """
        input_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs'
        }
        """


        if self.training:
            loss_dict = {
                "loss_segmentation": 0.,
                "loss_centermap": 0.,
                "loss_R": 0.
            }

            gt_bbx = self.getGTbbx(input_dict)

            ######################################################################
            # Implement PoseCNN's forward pass for training.               #
            #                                                                    #
            # Model should extract features, segment the objects, identify roi   #
            # object bounding boxes, and predict rotation and translations for   #
            # each roi box.                                                      #
            #                                                                    #
            # The training loss for semantic segmentation should be stored in    #
            # loss_dict["loss_segmentation"] and calculated using the            #
            # loss_cross_entropy(.) function.                                    #
            #                                                                    #
            # The training loss for translation should be stored in              #
            # loss_dict["loss_centermap"] using the L1loss function.             #
            #                                                                    #
            # The training loss for rotation should be stored in                 #
            # loss_dict["loss_R"] using the given loss_Rotation function.        #
            ######################################################################
            # Important: the rotation loss should be calculated only for regions
            # of interest that match with a ground truth object instance.
            # Note that the helper function, IOUselection, may be used for 
            # identifying the predicted regions of interest with acceptable IOU 
            # with the ground truth bounding boxes.
            # If no ROIs result from the selection, don't compute the loss_R
            
            feature1, feature2 = self.feat_extract(input_dict)
            probability, segmentation, bbx = self.seg_branch(feature1, feature2)
            output_bbxes = IOUselection(bbx, gt_bbx, self.iou_threshold)

            pred_centermaps = self.trans_branch(feature1, feature2) # this is the center map

            if len(output_bbxes)>1:
                quaternion = self.rot_branch(feature1, feature2, output_bbxes[:, :5])
                pred_Rs, label = self.estimateRotation(quaternion, output_bbxes)
                gt_Rs = self.gtRotation(output_bbxes, input_dict)
                loss_dict["loss_R"] = loss_Rotation(pred_Rs, gt_Rs, label.long(), self.models_pcd)


            loss_dict["loss_segmentation"] = loss_cross_entropy(probability, input_dict["label"])
            loss_dict["loss_centermap"] = torch.nn.functional.l1_loss(pred_centermaps, input_dict["centermaps"]) if len(bbx)>1 else 0.

            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################
            
            return loss_dict
        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():
                ######################################################################
                # Implement PoseCNN's forward pass for inference.              #
                ######################################################################
                feature1, feature2 = self.feat_extract(input_dict)
                probability, segmentation, bbx = self.seg_branch(feature1, feature2)

                centermaps = self.trans_branch(feature1, feature2)

                quaternion = self.rot_branch(feature1, feature2, bbx[:, :5].float())
                pred_Rs, label = self.estimateRotation(quaternion, bbx)
                
                pred_centers, pred_depths = self.HoughVoting(segmentation, centermaps)

                output_dict = self.generate_pose(pred_Rs, pred_centers, pred_depths, bbx.to(pred_Rs.device))
                ######################################################################
                #                            END OF YOUR CODE                        #
                ######################################################################

            return output_dict, segmentation

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        ## [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)
        
    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3, device=quaternion_map.device)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3, device=filter_bbx.device)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].detach().cpu().numpy()
            center = pred_centers[bs, obj_id - 1].detach().cpu().numpy()
            depth = pred_depths[bs, obj_id - 1].detach().cpu().numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict

    def HoughVoting(self, label, centermap, num_classes=10):
        """
        label [bs, 3, H, W]
        centermap [bs, 3*maxinstance, H, W]
        """
        batches, H, W = label.shape
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        xv, yv = np.meshgrid(x, y)
        xy = torch.from_numpy(np.array((xv, yv))).to(device=label.device, dtype=torch.float32)
        x_index = torch.from_numpy(x).to(device=label.device, dtype=torch.int32)
        centers = torch.zeros(batches, num_classes, 2, dtype=centermap.dtype, device=label.device)
        depths = torch.zeros(batches, num_classes, dtype=centermap.dtype, device=label.device)
        for bs in range(batches):
            for cls in range(1, num_classes + 1):
                if (label[bs] == cls).sum() >= _LABEL2MASK_THRESHOL:
                    pixel_location = xy[:2, label[bs] == cls]
                    pixel_direction = centermap[bs, (cls-1)*3:cls*3][:2, label[bs] == cls]
                    y_index = x_index.unsqueeze(dim=0) - pixel_location[0].unsqueeze(dim=1)
                    y_index = torch.round(pixel_location[1].unsqueeze(dim=1) + (pixel_direction[1]/pixel_direction[0]).unsqueeze(dim=1) * y_index).to(torch.int32)
                    mask = (y_index >= 0) * (y_index < H)
                    count = y_index * W + x_index.unsqueeze(dim=0)
                    center, inlier_num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
                    center_x, center_y = center % W, torch.div(center, W, rounding_mode='trunc')
                    if inlier_num > _HOUGHVOTING_NUM_INLIER:
                        centers[bs, cls - 1, 0], centers[bs, cls - 1, 1] = center_x, center_y
                        xyplane_dis = xy - torch.tensor([center_x, center_y])[:, None, None].to(device = label.device)
                        xyplane_direction = xyplane_dis/(xyplane_dis**2).sum(dim=0).sqrt()[None, :, :]
                        predict_direction = centermap[bs, (cls-1)*3:cls*3][:2]
                        inlier_mask = ((xyplane_direction * predict_direction).sum(dim=0).abs() >= _HOUGHVOTING_DIRECTION_INLIER) * label[bs] == cls
                        depths[bs, cls - 1] = centermap[bs, (cls-1)*3:cls*3][2, inlier_mask].mean()
        return centers, depths

def eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    ## image version vis
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb =  (rgb[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_dict[0],
        alpha = alpha
        )

def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * torch.log(scores + 1e-10), dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss

def loss_Rotation(pred_R, gt_R, label, model):
    """
    pred_R: a tensor [N, 3, 3]
    gt_R: a tensor [N, 3, 3]
    label: a tensor [N, ]
    model: a tensor [N_cls, 1024, 3]
    """
    device = pred_R.device
    models_pcd = model[label - 1].to(device)
    gt_points = models_pcd @ gt_R
    pred_points = models_pcd @ pred_R
    loss = ((pred_points - gt_points) ** 2).sum(dim=2).sqrt().mean()
    return loss


def IOUselection(pred_bbxes, gt_bbxes, threshold):
    """
        pred_bbx is N_pred_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        gt_bbx is gt_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        threshold : threshold of IOU for selection of predicted bbx
    """
    device = pred_bbxes.device
    output_bbxes = torch.empty((0, 6), device=device, dtype=torch.float)
    for pred_bbx in pred_bbxes:
        for gt_bbx in gt_bbxes:
            if pred_bbx[0] == gt_bbx[0] and pred_bbx[5] == gt_bbx[5]:
                iou = box_iou(pred_bbx[1:5].unsqueeze(dim=0), gt_bbx[1:5].unsqueeze(dim=0)).item()
                if iou > threshold:
                    output_bbxes = torch.cat((output_bbxes, pred_bbx.unsqueeze(dim=0)), dim=0)
    return output_bbxes

def getBbx(label, num_classes=10):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, num_classes - 1, steps = num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)
        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx

def getSegMask(label, num_classes=10):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        # label_repeat = label.view(bs, 1, H, W).repeat(1, num_classes, 1, 1).to(device)
        # label_target = torch.linspace(0, num_classes - 1, steps = num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = label != 0 # (1, H, W)
        return mask.squeeze() # tensor

def train_posecnn(data_dir, batch_size=2, num_classes=10, device="cuda"):
    # TODO set download to True if dataset doesn't exist
    train_dataset = PROPSPoseDataset(data_dir, "train", download=False) 

    print(f"Dataset sizes: train ({len(train_dataset)})")

    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    posecnn_model = PoseCNN(pretrained_backbone=vgg16, 
                            models_pcd=torch.tensor(train_dataset.models_pcd).to(device, dtype=torch.float32),
                            cam_intrinsic=train_dataset.cam_intrinsic,
                            num_classes=num_classes).to(device)
    posecnn_model.train()

    optimizer = torch.optim.Adam(posecnn_model.parameters(), lr=0.001,
                                betas=(0.9, 0.999))

    loss_history = []
    log_period = 50
    _iter = 0

    st_time = time.time()
    for epoch in range(10):
        train_loss = []
        dataloader.dataset.dataset_type = 'train'
        for batch in dataloader:
            for item in batch:
                batch[item] = batch[item].to(device)
            loss_dict = posecnn_model(batch)
            optimizer.zero_grad()
            total_loss = 0
            for loss in loss_dict:
                total_loss += loss_dict[loss]
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())
            
            if _iter % log_period == 0:
                loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
                for key, value in loss_dict.items():
                    loss_str += f"[{key}: {value:.3f}]"

                print(loss_str)
                loss_history.append(total_loss.item())
            _iter += 1

        print('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training finished' + f' , with mean training loss {np.array(train_loss).mean()}'))    

    torch.save(posecnn_model.state_dict(), os.path.join("checkpoints", "posecnn_model.pth"))

    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()

def eval_posecnn(data_dir, batch_size=2, num_classes=10, device="cuda"):
    val_dataset = PROPSPoseDataset(data_dir, "val")

    print(f"Dataset sizes: val ({len(val_dataset)})")

    dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    posecnn_model = PoseCNN(pretrained_backbone=vgg16, 
                            models_pcd=torch.tensor(val_dataset.models_pcd).to(device, dtype=torch.float32),
                            cam_intrinsic=val_dataset.cam_intrinsic,
                            num_classes=num_classes).to(device)
    
    model_dir = os.path.join("checkpoints", "posecnn_model.pth")
    print(f"Load PoseCNN model from {model_dir}")
    posecnn_model.load_state_dict(torch.load(model_dir))
    posecnn_model.eval()

    T_thresh = 5 # cm
    R_thresh = 5 # deg

    total =0
    correct = 0
    for batch in tqdm(dataloader):
        for item in batch:
            batch[item] = batch[item].to(device)
        pose_dict, segmentation = posecnn_model(batch)
        for bidx in range(batch_size):
            objs_visib = batch['objs_id'][bidx].cpu().tolist()
            objs_preds = sorted(list(pose_dict[bidx].keys()))
            for objidx, objs_id in enumerate(objs_visib):
                if objs_id==0:
                    continue

                total += 1
                if objs_id not in objs_preds:
                    continue
                RT_pred = pose_dict[bidx][objs_id]
                RT_true = batch['RTs'][bidx][objidx].cpu().numpy()

                # Translation error
                T_pred = RT_pred[:3,3]
                T_true = RT_true[:3,3]
                T_err = 100*np.linalg.norm(T_pred-T_true) # error in cm

                # Rotation error
                R_true = pyquaternion.Quaternion(matrix=RT_true[:3,:3],atol=1e-6)
                R_pred = pyquaternion.Quaternion(matrix=RT_pred[:3,:3],atol=1e-6)

                R_rel = R_pred * R_true.conjugate
                R_err = np.degrees(R_rel.angle)

                if T_err<T_thresh and R_err<R_thresh:
                    correct+=1

    print("Accuracy at 5°5cm:", correct/total)