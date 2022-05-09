#!/usr/bin/env python3
import os
import sys
import os.path
import time
from pathlib import Path
from urllib import request
from functools import reduce
import ctypes 
import cv2
import colorsys
import random
import numpy as np
from PIL import Image

base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.getenv("YOLO5l_OUTPUT_PATH")
model_path = os.getenv("YOLO_MODEL_PATH")


so_exe = ctypes.CDLL(os.path.join(output_path, "yolov5l.so"))


def preprocess_image(img, target_size=(640, 640)):
    # current shape [height, width]
    shape = img.shape[:2]

    # new shape [height, width]
    new_shape = target_size

    # Scale ratio (new / old)
    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = (int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio)))

    # wh padding
    delta_width, delta_height = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # divide padding into 2 sides
    delta_width /= 2
    delta_height /= 2

    # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # add border
    img = cv2.copyMakeBorder(img, 
                        int(round(delta_height - 0.1)), 
                        int(round(delta_height + 0.1)), 
                        int(round(delta_width - 0.1)), 
                        int(round(delta_width + 0.1)), 
                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    inputs = img.astype('float32')

    # normalize image
    inputs /= 255.

    # add batch axis if not present
    if inputs.ndim == 3:
        inputs = np.expand_dims(inputs, axis=0)

    return inputs
    

def run_inference(image_arr, is_save=False):
    image_arr = image_arr.flatten().astype(ctypes.c_float)
    if is_save:
        np.savetxt(os.path.join(output_path, "input_0.txt"), 
                image_arr.flatten().astype(np.float32), 
                fmt="%.17f", 
                delimiter=',', 
                encoding='utf-8')

    def struct_out(out):
        return (ctypes.c_float * reduce(lambda x, y: x * y, out))()

    outputs_shape = [(1, 25200, 85)]
    outputs = [struct_out(o) for o in outputs_shape]

    so_exe.model(image_arr.ctypes.data_as(ctypes.c_void_p), outputs[0])

    for out in outputs:
        _data = np.ctypeslib.as_array(out).astype(np.float32)
        _index = outputs.index(out)
        _shape = outputs_shape[_index]
        outputs[_index] = np.reshape(_data, _shape)
        if is_save:
            np.savetxt(os.path.join(output_path, f"output_{_shape[1]}.txt"), 
                _data.flatten(), 
                fmt="%.17f", 
                delimiter=',', 
                encoding='utf-8')

    return outputs


def xywh2xyxy(xywh):
    xyxy = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # top left x
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # top left y
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # bottom right x
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # bottom right y
    return xyxy


def detection_matrix(predictions, multi_label,conf_thres):

    # Compute conf = obj_conf * cls_conf
    predictions[:, 5:] *= predictions[:, 4:5]

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(predictions[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (predictions[:, 5:] > conf_thres).nonzero().T
        predictions = np.concatenate((box[i], predictions[i, j + 5, None], j[:, None].astype('float')), 1)

    # best class only
    else:
        j = np.expand_dims(predictions[:, 5:].argmax(axis=1), axis=1)
        conf = np.take_along_axis(predictions[:, 5:], j, axis=1)

        predictions = np.concatenate((box, conf, j.astype('float')), 1)[conf.reshape(-1) > conf_thres]

    return predictions


def get_iou(xyxy, order, areas, idx):
    x1, y1, x2, y2 = xyxy
    xx1 = np.maximum(x1[idx], x1[order[1:]])
    yy1 = np.maximum(y1[idx], y1[order[1:]])
    xx2 = np.minimum(x2[idx], x2[order[1:]])
    yy2 = np.minimum(y2[idx], y2[order[1:]])

    max_width = np.maximum(0.0, xx2 - xx1 + 1)
    max_height = np.maximum(0.0, yy2 - yy1 + 1)
    inter = max_width * max_height

    return inter / (areas[idx] + areas[order[1:]] - inter)


def nms_np(detections, scores, max_det, thresh):
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # get boxes with more ious first
    order = scores.argsort()[::-1]

    # final output boxes
    keep = []

    while order.size > 0 and len(keep) < max_det:
        # pick maxmum iou box
        i = order[0]
        keep.append(i)

        # get iou
        ovr = get_iou((x1, y1, x2, y2), order, areas, idx=i)

        # drop overlaping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def non_max_suppression_np(predictions):
    conf_thres, iou_thres = 0.25, 0.45
    multi_label, agnostic = False, False
    maximum_detections = 300
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    # number of classes > 1 (multiple labels per box (adds 0.5ms/img))
    multi_label &= (predictions.shape[2] - 5) > 1

    output = [np.zeros((0, 6))] * predictions.shape[0]
    confidences = predictions[..., 4] > conf_thres

    # image index, image inference
    for batch_index, prediction in enumerate(predictions):
        # confidence
        prediction = prediction[confidences[batch_index]]
        # print(prediction)

        # If none remain process next image
        if not prediction.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        prediction = detection_matrix(prediction, multi_label, conf_thres)

        # Check shape; # number of boxes
        if not prediction.shape[0]:  # no boxes
            continue

        # excess boxes
        if prediction.shape[0] > max_nms:
            prediction = prediction[np.argpartition(-prediction[:, 4], max_nms)[:max_nms]]

        # Batched NMS
        classes = prediction[:, 5:6] * (0 if agnostic else max_wh)
        indexes = nms_np(prediction[:, :4] + classes, prediction[:, 4], maximum_detections, iou_thres)

        # pick relevant boxes
        output[batch_index] = prediction[indexes, :]
    return output


def clip_coords(boxes, img_shape):
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2

    
def scale_coords(processed_shape, coords, original_shape, ratio_pad=None):
    # calculate from original_shape
    if ratio_pad is None:
        # gain  = old / new
        gain = min(processed_shape[0] / original_shape[0],
                   processed_shape[1] / original_shape[1])

        # wh padding
        pad = ((processed_shape[1] - original_shape[1] * gain) / 2,
               (processed_shape[0] - original_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # x padding
    coords[:, [0, 2]] -= pad[0]

    # y padding
    coords[:, [1, 3]] -= pad[1]

    coords[:, :4] /= gain
    clip_coords(coords, original_shape)
    return coords


def draw_bbox(image, output, rate, coco_class=None, is_show_obj=False):
    thickness = 3
    image_h, image_w, _ = image.shape

    num_classes = len(coco_class)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    msg = "HALO for Yolov5 "
    if not is_show_obj:
        msg += f", FPS: {rate}"
    cv2.putText(image, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    for *xyxy, conf, class_id in output:
        if class_id in coco_class:
            class_name=coco_class[class_id]
            bbox_color = colors[int(class_id)]

            x1, y1, x2, y2 = map(int, xyxy[:4])
            cv2.rectangle(image, (x1, y1), (x2, y2), color=bbox_color, thickness=bbox_thick)

            # show label 
            if is_show_obj:
                print(f"[{class_name.strip(',')}], pos:{xyxy} score:{round(conf, 3)}")

            title = f"{class_name.title()}: {round(conf, 2)}"
            scale = 0.5
            text_size = cv2.getTextSize(title, 0, fontScale=scale, thickness=bbox_thick//2)[0]
            top_left = (x1 - thickness + 1, y1 - text_size[1] - 20)
            bottom_right = (x1 + text_size[0] + 5, y1)
            cv2.rectangle(image, top_left, bottom_right, color=bbox_color, thickness=-1)
            cv2.putText(image, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


def load_class(coco_file_path):
    names = {}
    with open(coco_file_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

coco_class = load_class(os.path.join(os.path.dirname(base_path), "coco_classes.txt"))


def process_detection(original_image, is_show_obj=False):
    original_image_size = original_image.shape[:2]
    image = np.copy(original_image)

    image_arr = preprocess_image(image)
    infer_start_time = time.time()
    detections = run_inference(image_arr, is_save=is_show_obj)
    infer_finish_time = time.time()
    infer_rate = round(1 / (infer_finish_time - infer_start_time), 2)
    
    output = non_max_suppression_np(detections[0])[0]
    output[:, :4] = scale_coords(image_arr.shape[2:], output[:, :4], original_image.shape).round()
    image = draw_bbox(original_image, output, infer_rate, coco_class=coco_class, is_show_obj=is_show_obj)

    return image


if __name__ == "__main__":
    input_res = os.path.join(os.path.dirname(model_path), "zidane.jpg")
    if not os.path.exists(input_res):
        with request.urlopen("https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg") as req:
                with open(input_res, 'wb') as f:
                    f.write(req.read())

    input_res_path = Path(input_res)
    output_res = os.path.join(output_path, input_res_path.name.replace(".", "-halo."))
    print(f"[procesing] {input_res_path.name}")

    if not input_res_path.suffix.split('.')[-1] in ["mp4", "m4v"]:
        original_image = cv2.imread(input_res)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = process_detection(original_image, is_show_obj=True)
        image = Image.fromarray(image)
        image.save(output_res)
    else:
        cap = cv2.VideoCapture(input_res)
        if not cap.isOpened():
            print(f"Failure read video, {input_res}")
            sys.exit()

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_res, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height))

        start_time = time.time()
        while cap:
            retval, frame = cap.read()
            if not retval: break

            cnt_frame_start_time = time.time()
            process_frame = process_detection(frame)
            cnt_frame_finish_time = time.time()

            out.write(process_frame)
            
        cap.release()
        out.release()

        finish_time = time.time()
        print(f"total cost time: {round(finish_time - start_time, 3)}s")
    
