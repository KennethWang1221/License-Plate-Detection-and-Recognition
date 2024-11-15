#!/usr/bin/env python3

import argparse
import cv2
import os
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont

character_list = ["blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "云", "京", "冀", "吉", "学", "宁",
         "川", "挂", "新", "晋", "桂", "民", "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", "蒙", "藏", "警", "豫",
         "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", "黑", '领', '使', '澳', ]

def colors(i, bgr=False):
    hex_colors = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                  '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in hex_colors]
    n = len(palette)
    c = palette[int(i) % n]
    return (c[2], c[1], c[0]) if bgr else c

DEBUG = False

def frames_to_video(frames_folder, output_video_path, fps=25):
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in the specified folder.")
    
    # Read the first frame to get the dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()

def make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def concat_tail(args, datalist, file_name, inference_opt_tensor_path):

    input_data = []
    tensor80 = datalist[0].reshape(1,45,80,80).astype(np.float32)
    tensor40 = datalist[1].reshape(1,45,40,40).astype(np.float32)
    tensor20 = datalist[2].reshape(1,45,20,20).astype(np.float32)
    input_data.append(tensor80)
    input_data.append(tensor40)
    input_data.append(tensor20)

    nc = 2 # number of classes
    na = 3 # number of anchors
    no = nc+5+8 # number of outputs per anchor
    nl = 3 # number of detection layers
    stride = [8,16,32]
    anchors_list = [[4,5],[8,10],[13,16], [23,29],  [43,55],  [73,105],[146,217],  [231,300],  [335,433]]  # P3/8
    anchors = np.array(anchors_list, dtype=np.float32).reshape(nl, -1, 2)
    anchors = anchors.reshape(nl, 1, -1, 1, 1, 2)  # shape(nl,1,na,1,1,2)

    z = []
    for i in range(nl):
        tensor = input_data[i]
        bs, _, ny, nx = tensor.shape  
        tensor = tensor.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2).copy()

        grid = make_grid(nx,ny)
        y = np.zeros_like(tensor)
        class_range = list(range(5)) + list(range(13,13+nc))
        y[..., class_range] = sigmoid(tensor[..., class_range])
        y[..., 5:13] = tensor[..., 5:13]
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchors[i]  # wh

        y[..., 5:7]   = y[..., 5:7] *   anchors[i] + grid * stride[i] # landmark x1 y1
        y[..., 7:9]   = y[..., 7:9] *   anchors[i] + grid * stride[i]# landmark x2 y2
        y[..., 9:11]  = y[..., 9:11] *  anchors[i] + grid * stride[i]# landmark x3 y3
        y[..., 11:13] = y[..., 11:13] * anchors[i] + grid * stride[i]# landmark x4 y4
        
        z.append(y.reshape(bs, -1, no))

    concat = np.concatenate(z, axis=1)

    if DEBUG:
        save_path = os.path.join(inference_opt_tensor_path, '{}_concat.{}'.format(file_name,'tensor'))
        np.savetxt(save_path, concat.reshape(-1,1))

    return concat

def get_ignored_tokens():
    return [0]  # for ctc blank

def decodePlate(text_index, text_prob=None, is_remove_duplicate=False):
    """ convert text-index into text-label. """
    result_list = []
    ignored_tokens = get_ignored_tokens()
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        char_list = []
        conf_list = []
        for idx in range(len(text_index[batch_idx])):
            if text_index[batch_idx][idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                # only for predict
                if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                    continue
            char_list.append(character_list[int(text_index[batch_idx][idx])])
            if text_prob is not None:
                conf_list.append(text_prob[batch_idx][idx])
            else:
                conf_list.append(1)
        text = ''.join(char_list)
        result_list.append((text, np.mean(conf_list)))
    return result_list

def encode_images(image: np.ndarray, image_rec_prec_path, max_wh_ratio, target_shape, limited_max_width=160, limited_min_width=48):
    import math
    imgC = 3
    imgH, imgW = target_shape
    assert imgC == image.shape[2]
    max_wh_ratio = max(max_wh_ratio, imgW / imgH)
    imgW = int((imgH * max_wh_ratio))
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = image.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    if ratio_imgH > imgW:
        resized_w = imgW
    else:
        resized_w = int(ratio_imgH)
    
    resized_image = cv2.resize(image, (resized_w, imgH))
    padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padding_im[:, 0:resized_w, :] = resized_image
    if DEBUG:
        cv2.imwrite(image_rec_prec_path, padding_im)
    padding_im = padding_im.astype('float32')
    padding_im = (padding_im.transpose((2, 0, 1)) - 127.5) / 127.5

    return padding_im

def rec_pre_precessing(img, image_rec_prec_path, size=(48,160)): 

    h, w, _ = img.shape
    wh_ratio = w * 1.0 / h
    img = encode_images(img, image_rec_prec_path, wh_ratio, size)
    img = np.expand_dims(img, 0)

    return img

def get_plate_result(img, session_rec, image_rec_prec_path):
    img =rec_pre_precessing(img, image_rec_prec_path)
    result = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: img})

    if len(result)>0:
        prod = result[0]
        argmax = np.argmax(prod, axis=2)
        rmax = np.max(prod, axis=2)
        result = decodePlate(argmax, rmax, is_remove_duplicate=True)
        return result[0]
    else:
        return '',0.0

def letter_box(img, size=(640, 640)):
    h, w, c = img.shape
    ry, rx = size[0] / h, size[1] / w
    new_h, new_w = int(h * ry), int(w * rx)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
    return img, rx, ry, left, top

def preprocess(img0, file_path):

    ori_file_name = file_name =  os.path.splitext(os.path.basename(file_path))[0] 
    ori_image = img0.copy()
    dec_imgsz = [640,640]
    img, rx, ry, left, top = letter_box(img0, (dec_imgsz[0], dec_imgsz[1]))
    img_info = [rx, ry, left, top]

    img = img.transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    img = img.reshape(1, *img.shape)

    return img, img_info, ori_image, ori_file_name

def dec_inference(model, im):
    if len(im.shape) == 3:
        im = np.expand_dims(im, axis=0)  # expand for batch dim
    output_names = [output.name for output in model.get_outputs()]
    result = model.run(output_names, {model.get_inputs()[0].name: im})

    return result
    
def build_onnx_model(model_file):
    providers =  ['CPUExecutionProvider']
    session_detect = onnxruntime.InferenceSession(model_file, providers=providers )
    return session_detect
   
def xywh2xyxy(boxes):
    import copy
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh
 
def nms(boxes, iou_thresh): 
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(boxes, rx, ry, left, top):
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top

    boxes[:, [0, 2, 5, 7, 9, 11]] /= rx
    boxes[:, [1, 3, 6, 8, 10, 12]] /= ry
    return boxes

def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    points = points.astype(np.float32)
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  
    
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "./platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_result(orgimg,dict_list):
    result_str =""
    img_copy = orgimg.copy()
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=min(img_copy.shape[1],int(y-padding_h))
        rect_area[2]=max(0,int(rect_area[2]+padding_w))
        rect_area[3]=min(img_copy.shape[0],int(rect_area[3]+padding_h))

        height_area = result['roi_height']
        landmarks=result['landmarks']
        result = result['plate_no']
        result_str+=result+" "
        for i in range(4):  
            cv2.circle(img_copy, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(img_copy,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(255,255,0),2) 
        if len(result)>=7:
            img_copy=cv2ImgAddText(img_copy,result,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
    return img_copy

def rec_inference(args, dets, rec_model, im, img_info, im0s, ori_file_name, image_dect_results_path, image_rec_results_path, txt_results_path):
    file_name = os.path.splitext(os.path.basename(ori_file_name))[0]
    txt_path = os.path.join(txt_results_path, '{}.txt'.format(file_name))
    image_rec_path = os.path.join(image_rec_results_path, '{}_rec.png'.format(file_name)) 
    dict_list=[]
    log_info = []

    rx, ry, left,top = img_info[0],img_info[1],img_info[2],img_info[3]
    choice = dets[:, :, 4] > args['conf_thres']
    dets = dets[choice]
    dets[:, 13:15] *= dets[:, 4:5]
    box = dets[:, :4]
    boxes = xywh2xyxy(box)
    score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
    reserve_ = nms(output, args['iou_thres'])
    output = output[reserve_]
    outputs = restore_box(output, rx, ry, left, top)

    for index, out in enumerate(outputs):
        result_dict={}
        image_dect_path = os.path.join(image_dect_results_path, '{}_dect_{}.png'.format(file_name,index)) 
        rect = out[:4].astype(int) 
        score = out[4]
        land_marks = out[5:13].reshape(4, 2).astype(int) # layer_num = int(out[13])
        pad = get_rotate_crop_image(im0s, land_marks) 

        plate_code, rec_confidence = get_plate_result(pad, rec_model, image_rec_path)
        
        if plate_code == '':
            continue
        if len(plate_code) >= 7:
            dect_img = pad.copy()
            if DEBUG:
                cv2.imwrite(image_dect_path, dect_img)
            color=colors(1, True)
            result_dict['rect']=rect
            result_dict['landmarks']=land_marks.tolist()
            result_dict['plate_no']=plate_code
            result_dict['roi_height']=dect_img.shape[0]
            result_dict['plate_color']=color
            dict_list.append(result_dict)
            log_info.append([land_marks,str(plate_code),rect,score])

    with open(txt_path, 'a') as f:
        for result in log_info:
                log_message = (
                f"Plate Code: {result[1][:]}\n"
                f"Landmarks:\n {result[0]}\n"
                f"Rectangle: {np.array(result[2])}\n")
                f.write(log_message + "\n")
                print(log_message)

    final = draw_result(im0s,dict_list)
    cv2.imwrite(image_rec_path,final)

def main(**args):
    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])

    video_frames = os.path.join(args['opts_dir'], 'video_frames') 
    inference_opt_tensor_path = os.path.join(args['opts_dir'], 'inference_opt_tensor')
    image_dect_results_path = os.path.join(args['opts_dir'], 'image_dect_res') 
    image_rec_results_path = os.path.join(args['opts_dir'], 'image_rec_res') 
    txt_results_path = os.path.join(args['opts_dir'], 'txt_res') 
    inference_input_tensor_path = os.path.join(args['opts_dir'], 'inference_input_tensor')
    mp4_res = os.path.join(args['opts_dir'], '{}.mp4'.format(os.path.basename(args['opts_dir'])))

    if not os.path.exists(inference_opt_tensor_path):
        os.makedirs(inference_opt_tensor_path)
    if not os.path.exists(image_dect_results_path):
        os.makedirs(image_dect_results_path)
    if not os.path.exists(image_rec_results_path):
        os.makedirs(image_rec_results_path)
    if not os.path.exists(txt_results_path):
        os.makedirs(txt_results_path)
    if not os.path.exists(inference_input_tensor_path):
        os.makedirs(inference_input_tensor_path)
    if not os.path.exists(video_frames):
        os.makedirs(video_frames)

    dec_model = build_onnx_model(args['dec_model_file'])
    rec_model = build_onnx_model(args['rec_model_file'])

    cap = cv2.VideoCapture(args['video_path'])
    frame_count = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
#        if frame_count % frame_interval == 0:
        frame_filename = os.path.join(video_frames, f'frame_{frame_count:04d}.png')
        if DEBUG:
            cv2.imwrite(frame_filename, frame)

        im_preprocess, img_info, ori_image, ori_file_name = preprocess(frame, frame_filename)
        inference_res = dec_inference(dec_model, im_preprocess)
        output_con = concat_tail(args, inference_res, ori_file_name, inference_opt_tensor_path)
        rec_inference(args, output_con, rec_model, im_preprocess, img_info, ori_image, ori_file_name, image_dect_results_path, image_rec_results_path,txt_results_path)

        frame_count += 1

    cap.release()
    frames_to_video(image_rec_results_path, mp4_res, fps=int(frame_rate))

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Car License Detection + Recognition Demo')
    # Load file
    parser.add_argument("--dec_model_file", type=str,default="./models/dect_license.onnx", \
                        help='path to dec model')
    parser.add_argument("--rec_model_file", type=str,default="./models/rec_license.onnx", \
                        help='path to rec model')
    parser.add_argument("--video_path", type=str, default="demo.mp4", \
                        help='path to video')
    parser.add_argument("--conf_thres", type=float, default=0.3, \
                        help='conf thres')
    parser.add_argument("--iou_thres", type=float, default=0.45, \
                        help='iou thres')

    parser.add_argument("--opts_dir", type=str, default="./res", \
                        help='path of outputs files ')
    argspar = parser.parse_args()    

    print("\n### Test model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))