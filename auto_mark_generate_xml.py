from PIL import Image
import cv2
from yolo import YOLO
import xml.dom.minidom as DOC
import os
yolo = YOLO()  # 实例化模型


# ---------------------------------------------------------------------------- #
#  基于模型识别效果自动生成标注文件  --  适用于模型迭代 以yolo为例
#  如果需要快速使用的话，以下方法的常量无需修改也能直接使用，只需实例化自己的模型
#  需要在推理模型时返回的bbox 需要打包成[[left, top, right, bottom,  label],]格式即可。
# ---------------------------------------------------------------------------- #

def auto_generate_xml(img_name, coords, img_size, out_root_path):
    '''

    :param img_name: 对应图片名称
    :param coords: bbox [[left, top, right, bottom,  label],]
    :param img_size: 图片shape
    :param out_root_path: 输出xml目录
    :return:
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('data')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The WaterGauge Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('WaterGauge')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:
        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, img_name[:-4] + '.xml'), 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()

# ---------------------------------------------------------------------------- #
#  定义需要标注图片目录所在位置
#  遍历轮询推理，得到预标注的bbox，并生成对应的标注文件
# ---------------------------------------------------------------------------- #

images_dir = './mark_img'
images = sorted(os.listdir(images_dir))
for img in images:
    try:
        image = Image.open(os.path.join(images_dir, img))
        img_cv = cv2.imread(os.path.join(images_dir, img))
        img_size = img_cv.shape
        # ---------------------------------------------------------------------------- #
        #  注意此处返回的bbox_list:[[left, top, right, bottom, label],] label为分类的名称
        # ---------------------------------------------------------------------------- #
        r_image, bbox_list = yolo.detect_image(image)
    except Exception as e:
        print("该图片没有识别到物体，无法生成xml文件")
        continue
    else:
        auto_generate_xml(img, bbox_list, img_size, "./data_croped_xml")
