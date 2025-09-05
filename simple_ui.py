#!/usr/bin/env python3
"""
Simple Traffic Light Detector UI

This program provides a simple interface using OpenCV's GUI capabilities
to select between webcam and video file inputs for traffic light detection.
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import the detector class from the traffic_light_detector module
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir))

try:
    from traffic_light_detector import AutoTrafficLightDetector
except ImportError:
    print("Error: Could not import the AutoTrafficLightDetector class.")
    print("Make sure traffic_light_detector.py is in the same directory.")
    sys.exit(1)


class SimpleUI:

    def __init__(self):
        self.window_name = "Traffic Light Detector"
        self.menu_img = None
        self.menu_width = 800
        self.menu_height = 700
        self.selected_option = None
        self.detector = AutoTrafficLightDetector()

    def create_menu(self):
        """Create a simple menu using OpenCV"""
        # Create a blank image for the menu
        self.menu_img = np.ones((self.menu_height, self.menu_width, 3),
                                dtype=np.uint8) * 240  # Light gray background

        # Add title
        cv2.putText(self.menu_img, "Traffic Light Detector",
                    (self.menu_width // 2 - 180, 80), cv2.FONT_HERSHEY_DUPLEX,
                    1.5, (0, 100, 0), 2)

        # Add separator line
        cv2.line(self.menu_img, (50, 120), (self.menu_width - 50, 120),
                 (70, 70, 70), 2)

        # Add options
        cv2.putText(self.menu_img, "Select Input Source:", (100, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Option 1: Image File
        cv2.rectangle(self.menu_img, (150, 220), (650, 300), (0, 120, 0), 2)
        cv2.putText(self.menu_img, "1. Select Image File", (200, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Option 2: Webcam
        cv2.rectangle(self.menu_img, (150, 320), (650, 400), (0, 0, 120), 2)
        cv2.putText(self.menu_img, "2. Use Webcam", (200, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Option 3: Video File
        cv2.rectangle(self.menu_img, (150, 420), (650, 500), (120, 0, 120), 2)
        cv2.putText(self.menu_img, "3. Select Video File", (200, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Option 4: Exit
        cv2.rectangle(self.menu_img, (150, 520), (650, 600), (120, 0, 0), 2)
        cv2.putText(self.menu_img, "4. Exit", (200, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Add instructions
        cv2.putText(self.menu_img,
                    "Press the corresponding number key to select an option",
                    (150, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100),
                    1)

        return self.menu_img

    def show_menu(self):
        """Display the menu and handle user input"""
        menu_img = self.create_menu()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, menu_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):  # Image File
                self.selected_option = "image"
                break
            elif key == ord('2'):  # Webcam
                self.selected_option = "webcam"
                break
            elif key == ord('3'):  # Video File
                self.selected_option = "file"
                break
            elif key == ord('4') or key == 27:  # Exit (4 or ESC)
                self.selected_option = "exit"
                break

        cv2.destroyWindow(self.window_name)
        return self.selected_option

    def select_webcam(self):
        """Let the user select which webcam to use"""
        # Create a blank image for the menu
        select_img = np.ones(
            (300, 600, 3), dtype=np.uint8) * 240  # Light gray background

        cv2.putText(select_img, "Select Webcam ID:", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(select_img, "0 = Default Camera (usually built-in)",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "1, 2, etc. = External Cameras", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Press the number key for your selection",
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Press ESC to go back", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, select_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC to go back
                cv2.destroyWindow(self.window_name)
                return None
            elif ord('0') <= key <= ord('9'):
                webcam_id = key - ord('0')
                cv2.destroyWindow(self.window_name)
                return webcam_id

        return 0  # Default to webcam 0 if somehow we exit the loop

    def select_image_file(self):
        """
        Show instructions for selecting an image file
        
        Note: OpenCV doesn't provide a built-in file dialog, so we'll ask the user
        to enter the path in the terminal.
        """
        # Create a blank image for the instructions
        select_img = np.ones(
            (400, 700, 3), dtype=np.uint8) * 240  # Light gray background

        cv2.putText(select_img, "Select Image File:", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(select_img,
                    "Please enter the full path to your image file", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "in the terminal window.", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Supported formats: JPG, PNG, BMP, TIFF, etc.",
                    (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img,
                    "After entering the path, this window will close",
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img,
                    "and the detection will begin if the file exists.",
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Press any key to continue...", (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, select_img)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

        print("\nEnter the full path to your image file:")
        file_path = input().strip()

        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            time.sleep(2)  # Give user time to read the error
            return None

        return file_path

    def select_video_file(self):
        """
        Show instructions for selecting a video file
        
        Note: OpenCV doesn't provide a built-in file dialog, so we'll ask the user
        to enter the path in the terminal.
        """
        # Create a blank image for the instructions
        select_img = np.ones(
            (400, 700, 3), dtype=np.uint8) * 240  # Light gray background

        cv2.putText(select_img, "Select Video File:", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(select_img,
                    "Please enter the full path to your video file", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "in the terminal window.", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Supported formats: MP4, AVI, MOV, MKV, etc.",
                    (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img,
                    "After entering the path, this window will close",
                    (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img,
                    "and the detection will begin if the file exists.",
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.putText(select_img, "Press any key to continue...", (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, select_img)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)

        print("\nEnter the full path to your video file:")
        file_path = input().strip()

        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            time.sleep(2)  # Give user time to read the error
            return None

        return file_path

    def ask_save_output(self):
        """Ask the user if they want to save the processed output"""
        # Create a blank image for the menu
        save_img = np.ones(
            (300, 600, 3), dtype=np.uint8) * 240  # Light gray background

        cv2.putText(save_img, "Save processed video output?", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(save_img, "Y: Yes", (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 100, 0), 2)

        cv2.putText(save_img, "N: No", (50, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 100), 2)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, save_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y') or key == ord('Y'):
                cv2.destroyWindow(self.window_name)
                return True
            elif key == ord('n') or key == ord('N'):
                cv2.destroyWindow(self.window_name)
                return False

        return False  # Default to not saving

    def run(self):
        """Main method to run the UI"""
        while True:
            # Show main menu
            option = self.show_menu()

            if option == "exit":
                print("Exiting Traffic Light Detector.")
                break

            source = None
            save_output = False

            if option == "image":
                source = self.select_image_file()
                if source is not None:
                    save_output = self.ask_save_output()

            elif option == "webcam":
                source = self.select_webcam()
                if source is not None:
                    save_output = self.ask_save_output()

            elif option == "file":
                source = self.select_video_file()
                if source is not None:
                    save_output = self.ask_save_output()

            # Run the detector if source was selected
            if source is not None:
                print(
                    f"\nStarting traffic light detection with source: {source}"
                )
                print(f"Save output: {save_output}")

                try:
                    if option == "image":
                        # Process as image
                        result = self.detector.process_image_file(
                            source, save_output)
                        if result is not None:
                            # Display the result
                            cv2.imshow('Traffic Light Detection Result',
                                       result)
                            print("Press any key to close the window...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    else:
                        # Process as video/camera
                        self.detector.run(source, save_output)
                except Exception as e:
                    print(f"Error during detection: {e}")
                    # Show error message
                    error_img = np.ones((300, 600, 3), dtype=np.uint8) * 240
                    cv2.putText(error_img, "Error during detection:", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
                    cv2.putText(error_img, f"{str(e)}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    cv2.putText(error_img, "Press any key to continue...",
                                (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 1)
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(self.window_name, error_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(self.window_name)


def main():
    """Main function to start the application"""
    # Check OpenCV version
    if cv2.__version__ < '4.2.0':
        print(
            f"Warning: Your OpenCV version ({cv2.__version__}) might be too old."
        )
        print("This application was tested with OpenCV 4.5.0 or newer.")
        print("Continuing anyway, but you might encounter issues.")

    # Create and run UI
    ui = SimpleUI()
    ui.run()


if __name__ == "__main__":
    main()
