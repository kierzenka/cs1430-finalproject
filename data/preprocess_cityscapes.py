import os
import re

# run this inside the data folder 
# This gets all of the filepaths in gtFine_trainvaltest and leftImg8bit_trainvaltest

root_dir = './gtFine_trainvaltest'
with open('paths.txt', 'w') as f:
    for dir_, _, files in os.walk(root_dir):
        for file_name in files: 
            rel_dir = os.path.relpath(dir_, root_dir)
            rel_file = os.path.join(rel_dir, file_name)

            if re.match("(.*gtFine_labelTrainIds.png.*)", rel_file):
                image_path = 'leftImg8bit\\' + rel_file[7:39] + '_leftImg8bit.png'
                f.write(image_path + '\t' + rel_file + '\n')