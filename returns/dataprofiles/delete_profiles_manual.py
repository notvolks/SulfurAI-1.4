import os,shutil
current_dir_i = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir,  # up from Build to TrainingScript
    os.pardir,  # up to Sulfur
))
folder_path_output_dataprofiles = os.path.join(current_dir_i, 'RETURNS', 'dataprofiles', 'profiles')
if os.path.exists(folder_path_output_dataprofiles):
    if os.path.exists(folder_path_output_dataprofiles):
        for filename in os.listdir(folder_path_output_dataprofiles):
            file_path = os.path.join(folder_path_output_dataprofiles, filename)
            # Delete files
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            # Delete subfolders
            elif os.path.isdir(file_path):
                import shutil

                shutil.rmtree(file_path)
    print(f"Deleted folder: {folder_path_output_dataprofiles}")
else:
    print(f"Folder does not exist: {folder_path_output_dataprofiles}")

import time
time.sleep(100)