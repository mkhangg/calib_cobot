import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def find_pos(arr, query):
	id = -1

	for i in range(len(arr)):
		if arr[i] == query:
			id = i
			break
	return id

def xywh2center(boxs):
	centers = []

	for b in boxs:
		center = [round((b[0]+b[2])/2), round((b[1]+b[3])/2)]
		centers.append(center)

	return centers

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def load_label_for_gt(filename):
	data = np.loadtxt(filename)

	# read data.yaml
    # ['cone', 'cube', 'sphere']
    # [ 0    ,  1    ,  2      ]

	gt_labels = np.transpose(data)[0]

	data = (data[:,1:])

	boxes = []
	width, height = 640, 480

	for vec in data:		
		x_1 = (vec[0] - vec[2] / 2) * width
		y_1 = (vec[1] - vec[3] / 2) * width
		x_2 = (vec[0] + vec[2] / 2) * width
		y_2 = (vec[1] + vec[3] / 2) * width
		box = np.around(np.array([x_1, y_1, x_2, y_2]), 0).astype(int)
		boxes.append(box)
		
	boxes = np.array(boxes)

	return boxes, gt_labels


def load_label_for_preds(filename):
	raw_data = np.loadtxt(filename)
	confs = raw_data[:, -1]

	# read data.yaml
    # ['cone', 'cube', 'sphere']
    # [ 0    ,  1    ,  2      ]

	gt_labels = np.transpose(raw_data)[0]

	data = (raw_data[:,1:])

	boxes = []
	width, height = 640, 480

	for vec in data:		
		x_1 = (vec[0] - vec[2] / 2) * width
		y_1 = (vec[1] - vec[3] / 2) * width
		x_2 = (vec[0] + vec[2] / 2) * width
		y_2 = (vec[1] + vec[3] / 2) * width
		box = np.around(np.array([x_1, y_1, x_2, y_2]), 0).astype(int)
		boxes.append(box)
		
	boxes = np.array(boxes)

	return boxes, gt_labels, confs

# ============================================ #

# train_type = "freeze"
# train_type = "scratch"
train_type = "pretrain"
save_folder = f"v8_1000_{train_type}"
log_file_name = f"{save_folder}/conf_iou.txt"

log_file = open(log_file_name, "a")

names = ['cone', 'cube', 'sphere']

folder_path = "3d_shapes/test"
dir_list_images = os.listdir(folder_path + "/images")
dir_list_labels = os.listdir(folder_path + "/labels")

agg = sorted(dir_list_images) + sorted(dir_list_labels)
agg = np.array(agg)
agg = np.transpose(agg.reshape(2, int(len(agg)/2)))

for id in range(len(agg)):
    filename = agg[id][0]
    filename = filename[:-4]

    no_detections, no_misses, no_gts = 0, 0, 0

    source = f"3d_shapes/test/images/{filename}.jpg"

    gt_file_path = f"3d_shapes/test/labels/{filename}.txt"
    pred_file_path = f"runs/detect/predict_{train_type}/labels/{filename}.txt"

    gt_boxes, truth_classes = load_label_for_gt(gt_file_path)
    pred_boxes, pred_classes, probs = load_label_for_preds(pred_file_path)

    pred_boxes = pred_boxes.tolist()
    gt_boxes = gt_boxes.tolist()
    boxes_centers = xywh2center(pred_boxes)
    gt_boxes_centers = xywh2center(gt_boxes)

    orig_image = cv2.imread(source)
    color = [[255, 0, 0], [0, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 255]]
    info_box = f"Detected/Undetected: {len(pred_boxes)}/{len(gt_boxes) - len(pred_boxes)}"
    cv2.putText(orig_image, info_box,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                [255, 255, 255],
                2)  # line type


    no_detections += len(pred_boxes)
    no_misses += len(gt_boxes) - len(pred_boxes)
    no_gts += len(gt_boxes)

    if len(boxes_centers) > 0:
        all_centers = np.concatenate((np.array(boxes_centers), np.array(gt_boxes_centers)), axis=0)
    else:
        all_centers = np.array(gt_boxes_centers)

    pca = PCA(2)
    df = pca.fit_transform(all_centers)
    kmeans = KMeans(n_clusters= len(gt_boxes), random_state=0, n_init="auto")
    label = kmeans.fit_predict(df)

    labels_of_preds = label[0:len(boxes_centers)]
    labels_of_gt = label[-len(gt_boxes_centers):]

    labels_of_preds = labels_of_preds.tolist()
    labels_of_gt = labels_of_gt.tolist()

    for i in range(len(gt_boxes)):
        annotate = "" 
        gt_box = gt_boxes[i]
        corresponding_idx = find_pos(labels_of_preds, labels_of_gt[i])
        iou = 0

        if corresponding_idx == -1:
            cv2.rectangle(orig_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color[i], 1)

            iou = 0
            class_label = ""
            iou_label = ""

        else:
            box = pred_boxes[corresponding_idx]
            box = [round(b) for b in box]

            iou = compute_iou(box, gt_box)

            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color[i], 1)
            cv2.rectangle(orig_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color[i], 1)


            class_label = f"{names[int(pred_classes[corresponding_idx])]}: {probs[corresponding_idx]:.2f}"
            iou_label = " (" + str(round(iou, 2)) + ")" 

            log_file.write(f"{pred_classes[corresponding_idx]} {probs[corresponding_idx]:.4f} {iou:.4f}\n")
        
        annotate += class_label
        annotate += iou_label

        print(">>", annotate)

        cv2.putText(orig_image, annotate,
                    (gt_box[0] - 100, gt_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # font scale
                    color[i],
                    2)  # line type

    print("=================================================================================")
    cv2.imwrite(f"{save_folder}/out_iou_{id+1}.jpg", orig_image)

log_file.close()
