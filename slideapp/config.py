model_path=r"C:\Users\shimon.cohen\PycharmProjects\new_slidecore\model\output_model\resnet_epoch_17_0.924198.pt"
model_path=r"E:\medica_classifier\resnet_epoch_17_0.924198.pt"
inference_size=0
bad_dir=r"D:\medica_data\ReScan_bad"
good_dir=r""
out_dir=r"D:\medica_output"
classifer_tile_thr=0.4 # 0.4  #0.4           # 0.5
classifer_slide_thr=0.001                #  0.3
slide_img_down_sample=16 #2,4
max_working_days = 1    # How many days to work
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
tissue_z_thr = 0.5  # 0.5
tissue_anova_thr = 0.2

# work at lab
model_path="/mnt/medica/medica_classifier/resnet_epoch_17_0.924198.pt"
model_path="/mnt/medica/medica_classifier/resnet_epoch_59.pt"
# Next row best model so far
model_path="/mnt/medica/medica_classifier/resnet_epoch_219_98.862_v1.0.pt"
model_path="/mnt/medica/medica_classifier/resnet_epoch_214_98.3399_v1.1.pt"
model_path="/mnt/medica/medica_classifier/resnet_epoch_419_99.49_v1.1.pt"
model_path="/mnt/medica/medica_classifier/ensemble_model/ensemble_v1.1.pt"
model_path=r"E:\medica_classifier\resnet_epoch_13_arccos_0.9479103573.pt"

#input_dir="/mnt/medica/medica_data/ReScan_bad"
#input_dir="/home/shimon/hama-test"
input_file_exten='ndpi'         # "dcm" or "ndpi"
#input_dir="/home/shimon/Desktop/sectra-9.12.24/bad"
input_dir="/home/shimon/hama2/"
out_dir="/mnt/medica/medica_data/test_crone_folder_out"
input_dir=r"E:\medica_data\test_folder_for_train"
out_dir=r"E:\medica_data\test_folder_for_train_result"
#out_dir="/mnt/medica/medica_data/test_crone_folder_out_v1.1"
# input_dir="/mnt/medica/medica_data/to_debug"
# out_dir="/mnt/medica/medica_data/debug_out"
#input_dir="/home/shimon/Desktop/sectra-9.12.24"
#out_dir="/mnt/medica/medica_data/sectra-9.12.24"
#For test
tiles_working_dir="/mnt/medica/medica_data/tiles_dir"
# input_dir="/mnt/medica/medica_data/test_folder_for_train"
# out_dir="/mnt/medica/medica_data/for_test_folder_out"
downsample_slide=256        # If greater than 0 will generate a down sampled image for training slide classifier
work_list=[7]