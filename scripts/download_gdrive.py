#!/usr/bin/env python3
""" Download script for google drive shared links 

Thank you @turdus-merula and Andrew Hundt! 
https://stackoverflow.com/a/39225039/623735
"""
import sys
from nlpia2.google_drive import download_file_from_google_drive

def main():
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        file_id = sys.argv[1]  # shared google drive URL (link)
        destination = sys.argv[2]  # destination path on local disk
        download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
    main()
