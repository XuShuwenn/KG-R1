#!/usr/bin/env python3
"""
Download and prepare KGQA datasets from Google Drive
This script downloads compressed KGQA datasets from Google Drive, extracts them,
and organizes the data into the required directory structure
"""

import argparse
import os
import json
import requests
import tarfile
import shutil
import re
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse
import time

def get_gdrive_download_url(file_id: str) -> str:
    """Convert Google Drive file ID to direct download URL"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_file_from_gdrive(file_id: str, destination: str, filename: str) -> bool:
    """Download a file from Google Drive using file ID with proper handling for large files"""
    print(f"Downloading {filename}...")
    
    try:
        # Use a session for better connection handling
        session = requests.Session()
        
        # Initial download URL
        url = get_gdrive_download_url(file_id)
        
        # First request to check for virus scan warning
        print(f"Checking download for {filename}...")
        response = session.get(url)
        
        # If we get the virus scan warning page
        if response.status_code == 200 and 'virus scan warning' in response.text.lower():
            print(f"Handling virus scan warning for {filename}...")
            
            # Extract the form action URL and parameters from the HTML
            content = response.text
            
            # Look for the form action URL
            action_match = re.search(r'action="([^"]+)"', content)
            if action_match:
                action_url = action_match.group(1)
                print(f"Found form action URL: {action_url}")
                
                # Extract hidden form parameters
                params = {'id': file_id, 'export': 'download', 'confirm': 't'}
                
                # Look for additional parameters like uuid
                uuid_match = re.search(r'name="uuid" value="([^"]+)"', content)
                if uuid_match:
                    params['uuid'] = uuid_match.group(1)
                
                # Make the confirmed download request
                print(f"Making confirmed download request for {filename}...")
                response = session.get(action_url, params=params, stream=True)
                
        elif response.status_code == 200:
            # No virus scan warning, proceed with direct download
            response = session.get(url, stream=True)
        else:
            print(f"Failed to access {filename}. Status code: {response.status_code}")
            return False
        
        # Check if we got the file
        if response.status_code == 200:
            # Verify we're getting binary content, not HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print(f"Still getting HTML for {filename}. Trying alternative approach...")
                # Try the direct usercontent URL
                direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                response = session.get(direct_url, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                if total_size > 0:
                    print(f"Starting download of {filename} ({total_size / (1024*1024):.1f} MB)...")
                else:
                    print(f"Starting download of {filename} (size unknown)...")
                
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end='', flush=True)
                            else:
                                print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end='', flush=True)
                
                print(f"\n{filename} downloaded successfully! ({downloaded / (1024*1024):.1f} MB)")
                
                # Verify the downloaded file is not HTML
                if downloaded < 10000:  # Less than 10KB, might be an error page
                    with open(destination, 'r', encoding='utf-8', errors='ignore') as f:
                        content_check = f.read(1000)
                        if 'html' in content_check.lower() or '<!doctype' in content_check.lower():
                            print(f"Error: Downloaded file for {filename} appears to be HTML, not binary data")
                            return False
                
                return True
            else:
                print(f"Failed to download {filename}. Status code: {response.status_code}")
                return False
        else:
            print(f"Failed to download {filename}. Status code: {response.status_code}")
            if response.status_code == 403:
                print("This might be due to file permissions or rate limiting.")
            return False
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_and_cleanup(archive_path: str, extract_to: str, skip_cleanup: bool = False) -> bool:
    """Extract tar.gz file and remove the archive"""
    try:
        print(f"Extracting {os.path.basename(archive_path)}...")
        
        # Check if the file is actually a valid archive
        if not os.path.exists(archive_path):
            print(f"Error: Archive file {archive_path} does not exist")
            return False
        
        file_size = os.path.getsize(archive_path)
        print(f"Archive size: {file_size / (1024*1024):.1f} MB")
        
        # Check if the file is too small (might be an error page)
        if file_size < 1000:
            print(f"Warning: Archive file is very small ({file_size} bytes)")
            with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                if 'html' in content.lower() or '<' in content:
                    print(f"Error: Archive appears to be HTML content, not a valid archive")
                    return False
        
        # Try to open and extract the archive
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Get list of files in archive
                members = tar.getmembers()
                print(f"Archive contains {len(members)} files/directories")
                
                # Extract all files
                tar.extractall(path=extract_to)
                
                # Show some extracted files
                print("Extracted files:")
                for i, member in enumerate(members[:5]):  # Show first 5 files
                    print(f"  {member.name}")
                if len(members) > 5:
                    print(f"  ... and {len(members) - 5} more files")
                    
        except tarfile.ReadError as e:
            print(f"Error: Cannot read archive {archive_path}: {e}")
            print("The file might be corrupted or not a valid tar.gz file")
            return False
        except Exception as e:
            print(f"Error extracting archive: {e}")
            return False
        
        print(f"Extraction completed for {os.path.basename(archive_path)}")
        
        # Remove the compressed file to save space (unless skipped)
        if not skip_cleanup:
            os.remove(archive_path)
            print(f"Cleaned up {os.path.basename(archive_path)} (saved {file_size / (1024*1024):.1f} MB)")
        else:
            print(f"Keeping compressed file {os.path.basename(archive_path)}")
        
        return True
        
    except Exception as e:
        print(f"Error in extract_and_cleanup: {e}")
        return False

def download_dataset(dataset_name: str, file_id: str, save_path: str, skip_cleanup: bool = False) -> bool:
    """Download and extract a specific dataset"""
    archive_filename = f"{dataset_name}.tgz"
    archive_path = os.path.join(save_path, archive_filename)
    
    # Skip download if file already exists and is valid
    if os.path.exists(archive_path):
        file_size = os.path.getsize(archive_path)
        if file_size > 1000:  # More than 1KB, likely valid
            print(f"{archive_filename} already exists ({file_size / (1024*1024):.1f} MB), skipping download")
        else:
            print(f"{archive_filename} exists but is too small, re-downloading...")
            os.remove(archive_path)
    
    # Download the file if it doesn't exist or was removed
    if not os.path.exists(archive_path):
        if not download_file_from_gdrive(file_id, archive_path, archive_filename):
            return False
    
    # Extract and cleanup
    if not extract_and_cleanup(archive_path, save_path, skip_cleanup):
        return False
    
    return True

def download_all_datasets(save_path: str, datasets_to_download: List[str] = None, skip_cleanup: bool = False) -> Dict[str, bool]:
    """Download all or specified datasets from Google Drive"""
    
    # Google Drive file IDs for each dataset
    # Updated with actual file IDs from the Google Drive folder
    gdrive_file_ids = {
        'CWQ': '1ua7h88kJ6dECih6uumLeOIV9a3QNdP-g',           # CWQ.tgz (871.7 MB)
        'webqsp': '1KcIVAi4nf2uyflMOz5OSr54FOL2s2tAi',        # webqsp.tgz (136.4 MB)
        'metaqa-1hop': '1c_L9MfeYUCQh2Qq9gMbkcdrq72LTrNDi',   # metaqa-1hop.tgz
        'metaqa-2hop': '1SGE9_4VN8WQPxB0D_F3oDwTAJlgEK-vw',   # metaqa-2hop.tgz (914.6 MB)
        'metaqa-3hop': '1BQuz8ViA9xnIu60RJkqCm2ttKzGDM4Sx',   # metaqa-3hop.tgz (915.1 MB)
    }
    
    # Note: Some file IDs may be duplicated - please verify the URLs for each specific file
    duplicate_ids = set()
    for dataset, file_id in gdrive_file_ids.items():
        if list(gdrive_file_ids.values()).count(file_id) > 1:
            duplicate_ids.add(file_id)
    
    if duplicate_ids:
        print("WARNING: Some datasets share the same file ID:")
        for dataset, file_id in gdrive_file_ids.items():
            if file_id in duplicate_ids:
                print(f"  {dataset}: {file_id}")
        print("Please verify that each dataset has the correct unique file ID.")
        print("This may result in downloading the same file multiple times.")
    
    results = {}
    
    # If no specific datasets requested, download all
    if datasets_to_download is None:
        datasets_to_download = list(gdrive_file_ids.keys())
    
    for dataset_name in datasets_to_download:
        if dataset_name in gdrive_file_ids:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name} dataset")
            print(f"{'='*50}")
            
            success = download_dataset(dataset_name, gdrive_file_ids[dataset_name], save_path, skip_cleanup)
            results[dataset_name] = success
            
            if success:
                print(f"✓ {dataset_name} dataset processed successfully")
            else:
                print(f"✗ Failed to process {dataset_name} dataset")
                
            # Add a small delay between downloads to be respectful to the server
            if success:
                time.sleep(2)
        else:
            print(f"Warning: Unknown dataset '{dataset_name}' requested")
            results[dataset_name] = False
    
    return results

def organize_data_structure(save_path: str):
    """Organize the extracted data into the expected structure"""
    print("\nOrganizing data structure...")
    
    # Check what was actually extracted and provide a summary
    print("\nExamining extracted files...")
    for root, dirs, files in os.walk(save_path):
        level = root.replace(save_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")
    
    print("Data structure examination completed!")

def validate_downloads(save_path: str, downloaded_datasets: List[str]) -> Dict[str, bool]:
    """Validate that downloads were successful and files were extracted properly"""
    print("\nValidating downloads...")
    
    validation_results = {}
    
    for dataset in downloaded_datasets:
        print(f"Validating {dataset}...")
        
        # Check if dataset directory exists
        dataset_dir = os.path.join(save_path, dataset.lower())
        if os.path.exists(dataset_dir):
            files_count = sum(len(files) for _, _, files in os.walk(dataset_dir))
            dirs_count = sum(len(dirs) for _, dirs, _ in os.walk(dataset_dir))
            
            if files_count > 0:
                print(f"  ✓ {dataset}: {files_count} files, {dirs_count} directories")
                validation_results[dataset] = True
            else:
                print(f"  ✗ {dataset}: Directory exists but no files found")
                validation_results[dataset] = False
        else:
            # Check if files were extracted to the root directory
            possible_files = [f for f in os.listdir(save_path) 
                             if dataset.lower() in f.lower() and os.path.isfile(os.path.join(save_path, f))]
            possible_dirs = [d for d in os.listdir(save_path) 
                            if dataset.lower() in d.lower() and os.path.isdir(os.path.join(save_path, d))]
            
            if possible_files or possible_dirs:
                print(f"  ~ {dataset}: Files found but not in expected directory structure")
                print(f"    Files: {possible_files}")
                print(f"    Directories: {possible_dirs}")
                validation_results[dataset] = True
            else:
                print(f"  ✗ {dataset}: No files found")
                validation_results[dataset] = False
    
    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Download and prepare KGQA datasets from Google Drive.")
    parser.add_argument("--save_path", type=str, default="./data_kg", 
                        help="Directory to save all downloaded and processed data.")
    parser.add_argument("--datasets", nargs='+', 
                        choices=['CWQ', 'webqsp', 'metaqa-1hop', 'metaqa-2hop', 'metaqa-3hop'],
                        default=['CWQ', 'webqsp', 'metaqa-1hop', 'metaqa-2hop', 'metaqa-3hop'],
                        help="Specific datasets to download. Default: all datasets.")
    parser.add_argument("--skip_cleanup", action="store_true", 
                        help="Skip removal of compressed files after extraction.")

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Data will be saved to: {args.save_path}")

    # Download datasets
    print("\nStarting download process from Google Drive...")
    print("Note: Large files may take a while to download and extract.")
    
    results = download_all_datasets(args.save_path, args.datasets, args.skip_cleanup)
    
    # Get list of successfully downloaded datasets
    successful_datasets = [dataset for dataset, success in results.items() if success]
    
    # Organize data structure
    if successful_datasets:
        organize_data_structure(args.save_path)
        
        # Validate downloads
        validation_results = validate_downloads(args.save_path, successful_datasets)
    
    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    successful_downloads = []
    failed_downloads = []
    
    for dataset, success in results.items():
        if success:
            successful_downloads.append(dataset)
            print(f"✓ {dataset}")
        else:
            failed_downloads.append(dataset)
            print(f"✗ {dataset}")
    
    print(f"\nSuccessful downloads: {len(successful_downloads)}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"\nFailed datasets: {', '.join(failed_downloads)}")
        print("Please check your internet connection and Google Drive file IDs.")
        print("Run 'python get_gdrive_file_ids.py' for instructions on getting correct file IDs.")
    
    # Provide next steps
    if successful_downloads:
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Verify the extracted data structure matches your requirements")
        print("2. Check that all necessary files are present in each dataset directory")
        print("3. Update your training/inference scripts to use the new data paths")
        print(f"4. Data location: {args.save_path}")
    
    print(f"\nAll data saved under: {args.save_path}")
    print("Data preparation process completed!")

if __name__ == "__main__":
    main()
