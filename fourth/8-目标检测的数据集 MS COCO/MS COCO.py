''''''
'''
大规模的对象检测,分割,图像描述数据集

特点：
Object segmentation：目标级的分割（实例分割）
Recognition in context：上下文中的识别（图像情景识别）
Superpixel stuff segmentation：超像素分割
330K images (>200K labeled)：330K 图像（>200K 已经做好标记）
1.5 million object instances：150 万个对象实例
80 object categories：80 个目标类别
91 stuff categories： 91 个场景物体类别 （stuff中包含没有明确边界
的材料和对象，比如天空）
5 captions per image：每张图片 5 个情景描述（标题）
250,000 people with keypoints：250,000 人体的关键点标注

注意：80 object categories 是 91 stuff categories 的子集

80 object categories 是传统意义上的“物体”，

通常是可以单独识别和分割的具体对象。它们通常具有明确的边界，可以用边界框（boundingbox）
和分割掩码（segmentation mask）进行标注。例如：人（person）、自行车（bicycle）
这些物体类别在图像中通常是离散的，可以被独立标注和识别。

91 Stuff Categories 是“场景物体”或“背景物体”，

通常是一些没有明确边
界的区域，通常作为背景存在。它们不容易被单独识别，因为它们的边界
通常是连续的。这些类别在图像中通常覆盖大面积，且没有清晰的边界。
例如：草（grass）、天空（sky） 这些场景物体类别的标注通常用于场景
解析任务，例如场景分割（scene segmentation），而不是对象检测。

类别

'''
1