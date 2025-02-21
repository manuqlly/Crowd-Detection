# Crowd Detection

This project is designed to detect and visualize crowds in a video. It consists of three main scripts:

1. `detect.py`: Detects people in a video and records their positions frame by frame in a CSV file. This script uses the YOLOv8 model for detection and the SORT module for tracking. The SORT module needs to be installed from GitHub.

2. `detectcrowd.py`: Implements an algorithm to detect crowded groups of people. A crowd is defined as 3 or more people standing close together for at least 10 frames. The algorithm uses Euclidean distance to determine the closeness of people.

3. `visualize.py`: Visualizes the crowd data using Matplotlib and Seaborn. It generates a line plot showing the number of people over time and highlights the peak crowd sizes.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

Additionally, you need to install the SORT module from GitHub:

```sh
git clone https://github.com/abewley/sort.git
cd sort
"Install the requirement.txt inside the sort folder."
```

## Usage

1. **Detect People in Video**:
   Run detect.py to detect people in a video and save their positions frame by frame in a CSV file.

   ```sh
   python detect.py
   ```

2. **Detect Crowds**:
   Run detectcrowd.py to analyze the detected people and identify crowded groups.

   ```sh
   python detectcrowd.py
   ```

3. **Visualize Crowd Data**:
   Run visualize.py to visualize the crowd data using Matplotlib and Seaborn.

   ```sh
   python visualize.py
   ```

## License

This project is licensed under the MIT License.
