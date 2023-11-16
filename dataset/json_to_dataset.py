import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "批量处理json文件。"
        "文件夹中不能有其他文件存在！"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out

    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # 处理多个文件
    files = os.listdir(json_file)
    for i in range(0, len(files)):
        path = os.path.join(json_file, files[i])
        if os.path.isfile(path):
            try:
                data = json.load(open(path, encoding='UTF-8'))
            except:
                data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])

                try:
                    with open(imagePath, 'rb', encoding='UTF-8') as f:
                        imageData = f.read(encoding='UTF-8')
                        imageData = base64.b64encode(imageData).decode('utf-8')
                except:
                    with open(imagePath, 'rb') as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)

            label_name_to_value = {'_background_': 0}
            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value
            )

            label_names = [None] * (max(label_name_to_value.values()) + 1)
            for name, value in label_name_to_value.items():
                label_names[value] = name

            lbl_viz = imgviz.label2rgb(
                label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
            )

            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img', files[i]))
            utils.lblsave(osp.join(out_dir, 'mask'), files[i])
            # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
            # with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            #     for lbl_name in label_names:
            #         f.write(lbl_name + '\n')

            logger.info("Saved to: {}".format(out_dir))
