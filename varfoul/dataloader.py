
##<--- Omkar --->##
import os
import zipfile
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# Set up the downloader
local_directory = "SoccerNet"
mySNdl = SNdl(LocalDirectory=local_directory)

# Download the data
mySNdl.downloadDataTask(task="mvfouls", split=["train", "valid", "test", "challenge"], password="pass")

# Unzip the downloaded files
task_directory = os.path.join(local_directory, "mvfouls")
for split in ["train", "valid", "test", "challenge"]:
    zip_file = os.path.join(task_directory, f"{split}.zip")
    if os.path.exists(zip_file):
        # Create a new folder with the same name as the zip file
        extract_folder = os.path.join(task_directory, split)
        os.makedirs(extract_folder, exist_ok=True)

        # Extract the contents to the new folder
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Extracted {split}.zip to {extract_folder}")
    else:
        print(f"{split}.zip not found")

# Optionally, remove the zip files after extraction
for split in ["train", "valid", "test", "challenge"]:
    zip_file = os.path.join(task_directory, f"{split}.zip")
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"Removed {split}.zip")

##<--- Omkar --->##