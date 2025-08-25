import laspy

# 读取 LAS 文件
las = laspy.read(r"D:\user\Documents\ai\paper\1_process\dataSet\s3dis_pointNeXt\input\1-2(1_2)_导线.las")

# 查看点数量
print(f"点数量: {len(las.points)}")

# 查看所有属性字段
print("属性字段:", las.point_format.dimension_names)
# 显式列出所有属性字段（关键步骤）
print("所有属性字段:", list(las.point_format.dimension_names))
# 查看前5个点的坐标
print("坐标示例:")
print(las.x[:5])  # X坐标
print(las.y[:5])  # Y坐标
print(las.z[:5])  # Z坐标

# 查看其他属性（如强度、分类）
if 'intensity' in las.point_format.dimension_names:
    print("强度示例:", las.intensity[:5])
if 'classification' in las.point_format.dimension_names:
    print("分类示例:", las.classification[:5])