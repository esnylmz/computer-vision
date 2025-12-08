# computer-vision
## Computer Vision Project

The goal of this project is to develop a purely vision-based system that automatically transcribes a silent top-down video of a piano performance into a MIDI file. Unlike traditional music transcription methods that rely on audio signals, this project uses only visual information extracted from video frames to detect which piano keys are pressed and when, and then converts this information into symbolic musical notation.

- Methodology
--A fixed-camera, top-down video recording of a piano keyboard is used as input. In the initial step, the keyboard region is detected and geometrically normalized using perspective correction (homography). The rectified keyboard image is then segmented into individual key regions.

--For each video frame, motion and appearance changes inside each key region are analyzed using classical computer vision techniques such as frame differencing and intensity thresholding to identify key press events. Optionally, a lightweight CNN classifier may be trained to determine the pressed/unpressed state of each key for increased robustness.

--Detected key activation sequences are converted to note onset and offset times based on the video frame rate, and corresponding MIDI events are generated to construct a MIDI file representing the performance.
