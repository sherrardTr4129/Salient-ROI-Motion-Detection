# RBE526-Individual-Alg
The individual algorithm implementation assignment for RBE526

## usage
Run the script using the following command:
```bash
python3 computeSaliencyImage.py 
```

Note that this script attempts to create a VideoCapture object using /dev/video0. If you need to change that on your system, please do so. 

Four windows will open that display image streams taken from the various steps within the image processing pipeline (saliency map, object map, difference image, moving edges image). Once you are ready to close the program, click any of the windows and press 'q' on your keyboard. A list containing a running record of the total white pixels in the edge image is written to a CSV file in the script's local directory. From here, this data can be used to obtain a better threshold value to determine if the detected saliency ROI is moving between consecutive frames.

