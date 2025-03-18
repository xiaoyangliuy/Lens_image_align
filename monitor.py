import time
import os
import sys
import subprocess

folder_to_watch = r"C:\Users\lxiaoyang\Desktop\Zyla\Zyla"  # Change this to your target directory
# Path to the script to execute
script_to_run = r"C:\Research\OneDrive - Argonne National Laboratory\anl\github\Data_Analysis\Alignimg_1DGaufit_fwhm_figcsv_20250302.py"

# Store the existing folders
existing_folders = set(os.listdir(folder_to_watch))

def check_for_new_folders(check_time=2):
    global existing_folders
    while True:
        current_folders = set(os.listdir(folder_to_watch))
        new_folders = current_folders - existing_folders  # Find newly added folders

        for folder in new_folders:
            folder_path = os.path.join(folder_to_watch, folder)
            if os.path.isdir(folder_path):  # Ensure it's a directory
                print(f"New folder detected: {folder_path}")
                run_script(folder_path)  # Pass new folder to script

        existing_folders = current_folders
        time.sleep(check_time)  # Check every 2 seconds (adjust as needed)
        
def run_script(folder_path):
    """Execute the script and pass the new folder path as an argument."""
    try:
        subprocess.run(["python", script_to_run, folder_path], check=True)
        print(f"Successfully executed script for {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
if __name__ == "__main__":
    print(f"Monitoring {folder_to_watch} for new folders coming in")
    check_for_new_folders()       