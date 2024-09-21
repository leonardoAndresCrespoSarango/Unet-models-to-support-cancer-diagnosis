import splitfolders  # or import split_folders

input_folder = 'input_data_3channels/'
output_folder =  'split_dataset/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.6, .2, .2), group_prefix=None)