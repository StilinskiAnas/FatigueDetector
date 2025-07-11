# Fatigue Detector - Real-time Video Processing and Facial Fatigue Detection

## Overview

Fatigue Detector is a powerful desktop application developed by Anas Stilinski. It leverages Python and PyQt6 to perform real-time video processing and facial fatigue detection. The application offers a user-friendly interface for manipulating video feed properties and visualizing the results of facial action analysis.

## Features

- **Real-time Video Processing:** Apply various image processing techniques to the live video feed, such as grayscale conversion, blur, brightness adjustment, contrast enhancement, histogram equalization, and more.

- **Facial Fatigue Detection:** Utilize the facial action recognition module to analyze facial features, detect eyes, and assess fatigue levels based on eye aspect ratio (EAR) and mouth aspect ratio (MAR).

- **User-Configurable Settings:** Adjust image processing parameters using checkboxes and sliders for fine-tuning the visual output.

- **Data Logging:** Save the analyzed person data, including left and right eye aspect ratios, to a CSV file for further analysis.

## Installation

Ensure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

1. Run the application by executing the script:

```bash
python main.py
```

2. The application window will appear, displaying the live video feed and various options for image processing and facial fatigue detection.

3. Adjust the checkboxes and sliders to customize the image processing settings.

4. Observe the real-time changes in the displayed video feed.

5. The status bar at the bottom provides information about fatigue detection, and the table shows detailed person data.

6. The application automatically saves the detected person data to a CSV file every 5 seconds.

## Dependencies

- Python
- OpenCV
- PyQt6
- NumPy

## Contributors

- STILINSKI-ANAS

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Feel free to contribute, report issues, or suggest improvements!
