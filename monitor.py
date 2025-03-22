import time
import os
import sys
import subprocess

def check_for_new_folders(check_time=2,wait_time=10):
    global existing_folders
    while True:
        current_folders = set(os.listdir(folder_to_watch))
        new_folders = current_folders - existing_folders  # Find newly added folders

        for folder in new_folders:
            folder_path = os.path.join(folder_to_watch, folder)
            if os.path.isdir(folder_path):  # Ensure it's a directory
                print(f"New folder detected: {folder_path}")
                prev_files = set(os.listdir(folder_path))
                while True:
                    print(f'wait {wait_time} to check num of files in {folder_path}')
                    time.sleep(wait_time)  # check files every 10 seconds
                    current_files = set(os.listdir(folder_path))
                    if current_files != prev_files:
                        print(f"Taking data, now {len(os.listdir(folder_path))}")
                        prev_files = current_files
                        prev_files = set(os.listdir(folder_path))
                    else:
                        print(f"Complete for {folder_path}, total {os.listdir(folder_path)}")
                        run_script(folder_path)  # Pass new folder to script
                        break
        existing_folders = current_folders
        time.sleep(check_time)  # Check monitor folder every 2 seconds (adjust as needed)
        
def run_script(folder_path):
    """Execute the script and pass the new folder path as an argument."""
    try:
        subprocess.run(["python", script_to_run, folder_path], check=True)
        print(f"Successfully executed script for {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
if __name__ == "__main__":
    folder_to_watch = r"C:\Users\lxiaoyang\Desktop\Zyla\Zyla"  # Change this to your target directory
    # Path to the script to execute
    script_to_run = r"C:\Research\OneDrive - Argonne National Laboratory\anl\github\Lens_image_align\Alignimg_1DGaufit_fwhm_figcsv_20250302.py"

    # Store the existing folders
    existing_folders = set(os.listdir(folder_to_watch))
    print(f"Monitoring {folder_to_watch} for new folders coming in") 
    check_time = 5 #time to check monitor folder
    wait_time = 5 #time to check coming data files
    check_for_new_folders(check_time=check_time,wait_time=wait_time)  