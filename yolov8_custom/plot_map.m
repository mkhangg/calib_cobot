clc; clear; close all;

% Variables
width = 2.5;
no_models = 1; train_types = 3;
fig_rows = 1; fig_cols = 4;
map_file = "map_thr.txt";
line_styles = ["-*", "-o", "-square"];
colors = ["#7E2F8E", "#A2142F", "#EDB120", "#D95319", "#0072BD"];
folders = [["v8_1000_freeze", "v8_1000_pretrain", "v8_1000_scratch"]]; 

% Plot figure
figure('Name', 'Average Precision');
for i = 1 : no_models
    for j = 1 : train_types   
        path = folders(i, j);
        raw_data = readmatrix(path + "/" + map_file);
        [~, no_cols] = size(raw_data);
        for col = 2 : no_cols
            subplot(fig_rows, fig_cols, col - 1);
            plot(raw_data(:,col), linewidth = width);
%             plot(downsample(raw_data(:,col), 10), linewidth = width);
            hold on;
            grid on;
            xlabel('IoU Threshold'); ylabel('AP');
            ylim([0 1.05]); xlim([80, 100]);
        end 
    end
end
legend('Feature Extract. - YOLOv8-tiny', 'Fine Tuning - YOLOv8-tiny', 'Random Weights - YOLOv8-tiny')
