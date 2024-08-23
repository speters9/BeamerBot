# -*- coding: utf-8 -*-
"""
Utility for converting google drive sharable link to overleaf usable
"""

import re
import tkinter as tk
from tkinter import simpledialog


def convert_google_drive_link() -> str:
    """
    Converts a Google Drive shareable link into a direct download URL for use with Overleaf.

    Args:
        google_drive_url (str): The Google Drive shareable link.

    Returns:
        str: The direct download URL for Overleaf.
    """
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user for the Google Drive shareable link in a popup dialog
    google_drive_url = simpledialog.askstring("Input", "Please paste your Google Drive shareable link:")

    # Prompt the user for the Google Drive shareable link
    # google_drive_url = input("Please paste your Google Drive shareable link: ")

    # Regular expression to extract FILE_ID from the URL
    file_id_pattern = r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/view"
    file_id_match = re.search(file_id_pattern, google_drive_url)

    # Regular expression to extract RESOURCE_KEY if present in the URL
    resource_key_pattern = r"resourcekey=([a-zA-Z0-9_-]+)"
    resource_key_match = re.search(resource_key_pattern, google_drive_url)

    if file_id_match:
        file_id = file_id_match.group(1)
        resource_key = resource_key_match.group(1) if resource_key_match else None

        # Construct the direct download URL
        if resource_key:
            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}&resourcekey={resource_key}"
        else:
            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Create a new Tkinter window to display the URL
        output_window = tk.Toplevel(root)
        output_window.title("Direct Download URL")

        # Create an Entry widget with the direct download URL
        url_entry = tk.Entry(output_window, width=80)
        url_entry.insert(0, direct_download_url)
        url_entry.config(state='readonly')  # Make it readonly but still selectable
        url_entry.pack(pady=10)

        # Select the text in the Entry widget to make it easier to copy
        url_entry.selection_range(0, tk.END)

        # Add a label to instruct the user
        tk.Label(output_window, text="Copy the above URL for use in Overleaf.").pack(pady=5)

       # Add a button to close the window and stop the event loop
        def close():
            output_window.destroy()
            root.quit()  # Stop the event loop

        close_button = tk.Button(output_window, text="Close", command=close)
        close_button.pack(pady=10)

        # Start the Tkinter event loop to display the window
        root.mainloop()

        return direct_download_url
    else:
        tk.messagebox.showerror("Error", "Invalid Google Drive URL format.")
        root.destroy()
        raise ValueError("Invalid Google Drive URL format.")


# %%
# Example usage:
convert_google_drive_link()
