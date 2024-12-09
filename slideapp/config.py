model_path=r"C:\Users\shimon.cohen\PycharmProjects\new_slidecore\model\output_model\resnet_epoch_17_0.924198.pt"
model_path=r"E:\medica_classifier\resnet_epoch_17_0.924198.pt"
inference_size=0
bad_dir=r"D:\medica_data\ReScan_bad"
good_dir=r""
out_dir=r"D:\medica_output"
classifer_tile_thr=0.4 # 0.4  #0.4           # 0.5
classifer_slide_thr=0.001                #  0.3
slide_img_down_sample=2
tile_std_thr=30
write_tiles_into_out_dir = False         # For train purposes
bad_dir = r"E:\medica_data\ReScan_bad"
good_dir=r"E:\medica_data\ReScan_God"
# The root point at which results will be written to
input_dir=bad_dir
out_dir=r"E:\medica_data\results"
tiles_working_dir=r"E:\medica_data\tiles_dir"
run_flag = True     # Run mode
white_mean = 235.79956452889942
white_std = 1.663167324510908
white_conf = 3
tissue_mean=201.8
tissue_std=40.6
tissue_z_thr = 2  # 0.5
tissue_anova_thr = 0.5

# work at lab
model_path="/mnt/medica/medica_classifier/resnet_epoch_17_0.924198.pt"
model_path="/mnt/medica/medica_classifier/resnet_epoch_59.pt"
model_path="/mnt/medica/medica_classifier/resnet_epoch_219_98.862.pt"
#out_dir="/mnt/medica/medica_data/test_folder_out"
#input_dir="/mnt/medica/medica_data/ReScan_bad"
#input_dir="/home/shimon/hama-test"
#input_dir="/home/shimon/hama2/"
#For test
tiles_working_dir="/mnt/medica/medica_data/tiles_dir"
input_dir="/mnt/medica/medica_data/test_folder_for_train"
out_dir="/mnt/medica/medica_data/for_test_folder_out"
downsample_slide=256        # If greater than 0 will generate a down sampled image for training slide classifier