import numpy as np
import torch

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)   # Shape: [1,1, 5, 5], filled with zeros
    max_mask = scores == max_pool(scores)  #对整张图进行最大池化，得到最大值的位置，并赋值为True，其余位置为False，此示例即为0.9对应的位置
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)  #torch.where(condition, x, y) condition (bool型张量) ：当condition为真，返回x的值，否则返回y的值
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))    #|按位或，只要对应的两个二进制位中有一个为 1，结果位就为 1，否则为 0。  &按位与， 只有对应的两个二进制位都为 1 时，结果位才为 1，否则为 0
    return torch.where(max_mask, scores, zeros)

if __name__ == '__main__':
    # data = torch.tensor([
    # [0.1, 0.3, 0.2, 0.4, 0.5],
    # [0.2, 0.5, 0.4, 0.3, 0.1],
    # [0.6, 0.4, 0.5, 0.7, 0.3],
    # [0.2, 0.7, 0.6, 0.8, 0.4],
    # [0.5, 0.6, 0.7, 0.9, 0.8]
    # ], dtype=torch.float32)

    data = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.8, 0.7, 0.5],
    [0.6, 0.4, 0.9]
    ], dtype=torch.float32)

    # Convert data to a PyTorch tensor and add batch and channel dimensions
    scores = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 1, 5, 5]
    
    # Apply NMS
    result = simple_nms(scores, 1)  # Shape: [1, 1, 5, 5]
    
    # Convert the result back to NumPy for printing
    # result_np = result.squeeze() # Shape: [5, 5]
    
    # print(result_np)

    keypoints = [
            torch.nonzero(s > 0.001)
            for s in result]    #获取张量中所有非零元素的索引
    for s, k in zip(result, keypoints):
        print(s, k)
        print(k.t())
        print(tuple(k.t()))
        print(s[tuple(k.t())])
    scores = [s[tuple(k.t())] for s, k in zip(result, keypoints)]    # zip打包为元组的列表，tuple(k.t())将k的行和列互换

    print(keypoints)
