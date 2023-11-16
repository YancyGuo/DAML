from PIL import Image

import numpy as np
import os
import shutil  # 导入模块


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [width, height]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=1))[0]
    vertical_indicies = np.where(np.any(m, axis=0))[0]
    if horizontal_indicies.shape[0]:

        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    y2 = mask.shape[0] - 1 if y2 >= mask.shape[0] else y2
    x2 = mask.shape[1] - 1 if x2 >= mask.shape[1] else x2
    boxes = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def crop(img_path, mask_path, croped_img_dir):  # 定义函数名称
    filenames = os.listdir(img_path)
    for filename in filenames:
        # 加载原始图片
        img = Image.open(os.path.join(img_path, filename))
        mask = Image.open(os.path.join(mask_path, filename))
        # 获取坐标
        box = extract_bboxes(np.asarray(mask))
        img2 = img.crop(box)
        img2.save(os.path.join(croped_img_dir, filename))
        print(filename, "croped")  # 输出提示


if __name__ == '__main__':
    img_path = r'D:\data\json\8\img'
    mask_path = r'D:\data\json\8\mask'
    croped_img_dir = r'D:\data\ROI\8'
    if not os.path.exists(croped_img_dir):
        os.makedirs(croped_img_dir)
    crop(img_path, mask_path, croped_img_dir)
