import numpy as np

def compute_ap(data, no_ground_truths):

    total_gts = no_ground_truths
    cumulative_TP, cumulative_FP = [], []
    precision_arr, recall_arr = [], []

    for i in range(len(data)):
        if data[i][1] == 1:
            cumulative_TP.append(1)
            cumulative_FP.append(0)
        else:
            cumulative_TP.append(0)
            cumulative_FP.append(1)

    cumulative_TP = np.cumsum(np.array(cumulative_TP))
    cumulative_FP = np.cumsum(np.array(cumulative_FP))

    for i in range(len(data)):
        precision_arr.append(cumulative_TP[i]/(cumulative_TP[i] + cumulative_FP[i])) 
        if cumulative_TP[i] <= total_gts:
            recall_arr.append(cumulative_TP[i]/total_gts)
        else:
            recall_arr.append(cumulative_TP[i]/len(data))

    precision_arr = np.concatenate([[0.0], precision_arr, [0.0]])
    
    for i in range(len(precision_arr) - 1, 0, -1):
        precision_arr[i - 1] = np.maximum(precision_arr[i - 1], precision_arr[i])

    recall_arr = np.concatenate([[0.0], recall_arr, [1.0]])
    changing_points = np.where(recall_arr[1:] != recall_arr[:-1])[0]
    areas = (recall_arr[changing_points + 1] - recall_arr[changing_points]) * precision_arr[changing_points + 1]

    return areas.sum()

def compute_map(ap_arr):
    map = np.sum(ap_arr)/len(ap_arr)
    return map

# ============================================ 

# train_type = "freeze"
# train_type = "scratch"
train_type = "pretrain"
save_folder = f"v8_1000_{train_type}"
log_file_name = f"{save_folder}/conf_iou.txt"

threshold_step = 0.01
# log_file_path = new_folder_path + "/" + log_file_name

inference_data = np.loadtxt(log_file_name)
iou_threshold_list = np.arange(0.01, 1.00, threshold_step)

target_classes = [0, 1, 2]
counts = [30, 58, 60]
classes_names = ["cone", "cube", "sphere"]

map_file_name = f"{save_folder}/map_thr.txt"
map_file = open(map_file_name, "w")

for j in range(len(iou_threshold_list)):
    classes_aps = []
    iou_threshold = iou_threshold_list[j]

    for i in range(len(target_classes)):
        target_class = target_classes[i]

        target_class_arr = []
        for row in range(len(inference_data)):
            if int(inference_data[row][0]) == target_class:
                target_class_arr.append(inference_data[row])
        
        target_class_arr = np.delete(np.array(target_class_arr), 0, 1)
        target_class_arr = np.flip(target_class_arr[target_class_arr[:, 0].argsort()], axis=0)

        for row in range(len(target_class_arr)):
            if target_class_arr[row][1] >= iou_threshold:
                target_class_arr[row][1] = 1
            else:
                target_class_arr[row][1] = 0

        ap_class = compute_ap(target_class_arr, counts[i])
        classes_aps.append(ap_class)
        print(f">> AP of {classes_names[i]} = {ap_class:.4f}", )

    print(f">> mAP at {iou_threshold:.2f} = {compute_map(classes_aps):.4f}")

    map_file.write(f"{iou_threshold:.2f} {classes_aps[0]:.4f} {classes_aps[1]:.4f} {classes_aps[2]:.4f} {compute_map(classes_aps):.4f}\n")

    print("=================================================================================")

map_file.close()
