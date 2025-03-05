# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)   #获得一个全0张量，shape与scores相同
    max_mask = scores == max_pool(scores)  #得到初始极大值掩码。若某点的值等于池化窗口内的最大值，则标记为候选点，即对整张图进行最大池化，得到最大值的位置，并赋值为True，其余位置为False，注意padding
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0   #将当前 max_mask 转换为浮点类型并池化，得到抑制掩码。池化操作会将每个 True 点周围 nms_radius 内的区域标记为需抑制的区域
        supp_scores = torch.where(supp_mask, zeros, scores) #将抑制掩码应用到 scores 上，得到抑制后的分数。即如果某点在抑制掩码中为True，则该点的分数被置为0
        new_max_mask = supp_scores == max_pool(supp_scores)  #得到新的极大值掩码。即对抑制后的分数进行最大池化，得到新的极大值的位置，并赋值为True，其余位置为False
        max_mask = max_mask | (new_max_mask & (~supp_mask)) #更新极大值掩码。即如果某点在新的极大值掩码中为True，且在抑制掩码中为False，则该点在极大值掩码中标记为True
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]  #将keypoints归一化到[0,1]之间，使得与descriptors的值在同一维度上
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(        
        #torch.nn.functional.grid_sample将input图像特征体通过grid映射到output图像特征体上,底层使用双线性插值，interpolate是规则采样（uniform)，
        # 但是grid_sample的转换方式，内部采点的方式并不是规则的，是一种更为灵活的方式。
        # input  B C Hi Wi
        # grid   B Hg Wg 2
        # output B C Hg Wg
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)  #torch.Size([1, 256, 1, 651])
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)    #torch.Size([1, 256, 651])
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # print('data_img', data['image'].shape)
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        # print('0.5x= ', x.shape)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        # print('0.25x= ', x.shape)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        # print('0.125x= ', x.shape)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # print('encode_x= ', x.shape)

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        # print('cpa', cPa.shape)
        scores = self.convPb(cPa)
        # print('scores', scores.shape)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]   # 1, 64, 60 80
        b, _, h, w = scores.shape
        # print('b, _, h, w= ', b, _, h, w)
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)  # torch.Size([1, 60, 80, 8, 8])
        # print('scores reshape0= ', scores.shape)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)   #torch.Size([1, 480, 640])
        print('scores reshape1= ', scores.shape)
        scores = simple_nms(scores, self.config['nms_radius'])
        # print(scores)
        # print('scores nms= ', scores.shape)

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]    #获取张量中所有非零元素对应的行、列索引，并以坐标形式返回
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y) (651, 2)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]  #torch.flip 第一个参数是输入，第二个参数是输入的第几维度，按照维度对输入进行翻转

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1) # p可以选择l2 和l1  torch.Size([1, 256, 60, 80])
        # print('descriptors', descriptors.shape)

        # Extract descriptors, 提取关键点描述符
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,   #(651, 2)
            'scores': scores,        #(651, )
            'descriptors': descriptors,  #(256, 651)
        }
