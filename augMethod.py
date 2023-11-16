from imgaug import augmenters as iaa
import numpy as np
class MyAugMethod():

    def __init__(self):
        self.seq = iaa.Sequential()


    # 定义增强的方法
    # def aug_method(self):
    #     # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    #     # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    #     # image.
    #     sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    #
    #     # # Define our sequence of augmentation steps that will be applied to every image.
    #     # self.seq = iaa.Sequential([
    #     #     # Execute 0 to 5 of the following (less important) augmenters per
    #     #     # image. Don't execute all of them, as that would often be way too
    #     #     # strong.
    #     #     #
    #     #     iaa.SomeOf((0, 5),
    #     #                [
    #     #                    # 将图像进行超分辨率，每幅图采样20到200个像素，
    #     #                    # 替换其中的一些值，但不会使用平均值来替换所有的超像素
    #     #                    sometimes(
    #     #                        iaa.Superpixels(
    #     #                            p_replace=(0, 1.0),
    #     #                            n_segments=(20, 200)
    #     #                        )
    #     #                    ),
    #     #
    #     #                    # 使用不同的模糊方法来对图像进行模糊处理
    #     #                    # 高斯滤波
    #     #                    # 均值滤波
    #     #                    # 中值滤波
    #     #                    iaa.OneOf([
    #     #                        iaa.GaussianBlur((0, 3.0)),
    #     #                        iaa.AverageBlur(k=(2, 7)),
    #     #                        iaa.MedianBlur(k=(3, 11)),
    #     #                    ]),
    #     #
    #     #                    # 对图像进行锐化处理，alpha表示锐化程度
    #     #                    # Sharpen each image, overlay the result with the original
    #     #                    # image using an alpha between 0 (no sharpening) and 1
    #     #                    # (full sharpening effect).
    #     #                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    #     #
    #     #                    # Same as sharpen, but for an embossing effect
    #     #                    # 与sharpen锐化效果类似，但是浮雕效果
    #     #                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    #     #
    #     #                    # 添加高斯噪声
    #     #                    # Add gaussian noise to some images.
    #     #                    # In 50% of these cases, the noise is randomly sampled per
    #     #                    # channel and pixel.
    #     #                    # In the other 50% of all cases it is sampled once per
    #     #                    # pixel (i.e. brightness change).
    #     #                    iaa.AdditiveGaussianNoise(
    #     #                        loc=0, scale=(0.0, 0.05 * 255)
    #     #                    ),
    #     #
    #     #                    # Add a value of -10 to 10 to each pixel. 每个像素增加（-10,10）之间的像素值
    #     #                    iaa.Add((-10, 10), per_channel=0.5),
    #     #
    #     #                    # 将-40到40之间的随机值添加到图像中，每个值按像素采样
    #     #                    iaa.AddElementwise((-40, 40)),
    #     #
    #     #                    # 改变图像亮度（原值的50-150%）
    #     #                    iaa.Multiply((0.5, 1.5)),
    #     #
    #     #                    # # 将每个像素乘以0.5到1.5之间的随机值.
    #     #                    # iaa.MultiplyElementwise((0.5, 1.5)),
    #     #
    #     #                    # Improve or worsen the contrast of images.
    #     #                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    #     #
    #     #                    # crop images from each side by 0 to 16px (randomly chosen)
    #     #                    # crop some of the images by 0-10% of their height/width
    #     #                    iaa.Crop(px=(0, 80)),
    #     #
    #     #                    # 0.5 is the probability, horizontally flip 50% of the images
    #     #                    iaa.Fliplr(0.5),
    #     #
    #     #                    iaa.FrequencyNoiseAlpha(
    #     #                        first=iaa.Affine(
    #     #                            rotate=(-40, 40),
    #     #                            translate_px={"x": (-4, 4), "y": (-4, 4)}
    #     #                        ),
    #     #                        second=iaa.AddToHueAndSaturation((-40, 40)),
    #     #                        per_channel=0.5
    #     #                    )
    #     #
    #     #                ],
    #     #                # 按随机顺序进行上述所有扩充
    #     #                random_order=True
    #     #                )
    #     #
    #     # ], random_order=True)
    #
    #     # 增强函数

    def aug_method(self):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
        # image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        sometimes_for_all_images = lambda aug: iaa.Sometimes(0.2, aug)

        # # Define our sequence of augmentation steps that will be applied to every image.
        self.seq = iaa.Sequential([
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            # 0.5 is the probability, horizontally flip 50% of the image
            sometimes_for_all_images(
            iaa.SomeOf((0, 5),
                       [
                           # 使用不同的模糊方法来对图像进行模糊处理
                           # 高斯滤波
                           # 均值滤波
                           # 中值滤波
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 5)),
                           ]),

                           # 对图像进行锐化处理，alpha表示锐化程度
                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           iaa.Sharpen(alpha=(0, 0.8), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect
                           # 与sharpen锐化效果类似，但是浮雕效果
                           iaa.Emboss(alpha=(0, 0.8), strength=(0, 2.0)),

                           # 添加高斯噪声
                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255)
                           ),

                           # Add a value of -10 to 10 to each pixel. 每个像素增加（-10,10）之间的像素值
                           iaa.Add((-10, 10), per_channel=0.5),

                           # 将-40到40之间的随机值添加到图像中，每个值按像素采样
                           iaa.AddElementwise((-40, 40)),

                           # 改变图像亮度（原值的50-150%）
                           iaa.Multiply((0.5, 1.5)),

                           # # 将每个像素乘以0.5到1.5之间的随机值.
                           iaa.MultiplyElementwise((0.5, 1.5)),

                           # Improve or worsen the contrast of images.
                           iaa.LinearContrast((0.5, 2.0)),

                           # crop images from each side by 0 to 16px (randomly chosen)
                           # crop some of the images by 0-10% of their height/width
                           iaa.Crop(px=(0, 80)),
                           iaa.Fliplr(0.5),

                           # iaa.Affine(
                           #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                           #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                           #     rotate=(-25, 25),
                           #     shear=(-8, 8)
                           # ),
                           iaa.FrequencyNoiseAlpha(
                                                      first=iaa.Affine(
                                                          rotate=(-40, 40),
                                                          translate_px={"x": (-4, 4), "y": (-4, 4)}
                                                      ),
                                                      second=iaa.AddToHueAndSaturation((-40, 40)),
                                                      per_channel=0.5
                                                  )

                       ],
                       # 按随机顺序进行上述所有扩充
                       random_order=True
                       )
            )
        ], random_order=True)

        # 增强函数

    def aug_data(self, imglist):
        # 实例化增强方法
        self.aug_method()
        # 这句是后加的，便于函数接受到正确的类型
        imglist = np.array(imglist).astype(np.uint8)
        # 对文件夹中的图片进行增强操作

        images_aug = self.seq.augment_images(imglist)
        return images_aug