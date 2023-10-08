## Training the models
1. navigate to the main directory of the repository
```
cd ~/<your_local_repo_path>/touch2image/reskin/
```
2. run
```
python models/trainae.py --model_id <your_id> --infer_type <"image" or "tactile"> --encode_image <boolean> --encode_tactile <boolean>
```

[optional] If you wish to customize the training process:

1. navigate to the models configuration directory
```
cd ~/<your_local_repo_path>/touch2image/reskin/models/conf
```
2. open `trainae.yaml` file 
3. adapt the controls as you wish

4. navigate to the main directory of the repository
```
cd ~/<your_local_repo_path>/touch2image/reskin/
```
5. run
```
python models/trainae.py
```

## Data Collection
1. navigate to the main directory of the repository
```
cd ~/<your_local_repo_path>/touch2image/reskin/
```
2. run
```
python data_collection/collect_data.py
```

If you wish to customize the collection process:
1. navigate to the data_collection configuration directory
```
cd ~/<your_local_repo_path>/touch2image/reskin/data_collection/conf
```
2. open `collection.yaml` file 
3. adapt the controls as you wish

## Object Recognition
### PCDs rendering
1. navigate to the task configuration directory
```
cd ~/<your_local_repo_path>/touch2image/reskin/task/conf
```
2. open `save_pcds.yaml` file 
3. adapt the controls as you wish
4. run
```
python task/save_pcds_extra_views.py
```

### Online recognition
1. navigate to the task configuration directory
```
cd ~/<your_local_repo_path>/touch2image/reskin/task/conf
```
2. open `online.yaml` file 
3. adapt the controls as you wish
4. run
```
python task/recognize_object_online.py
```
# Reskin Sensor Library
This is a python library to interface with [ReSkin](https://openreview.net/forum?id=87_OJU4sw3V) sensors. We provide two classes for interfacing with [ReSkin](https://openreview.net/forum?id=87_OJU4sw3V). The `ReSkinBase` class is good for standalone data collection: it blocks code execution while data is being collected. The `ReSkinProcess` class can be used for non-blocking background data collection. Data can be buffered in the background while you run the rest of your code.  

## Installation

1. Clone this repository using 
```
$ git clone https://github.com/raunaqbhirangi/reskin_sensor.git
```
2. Install dependencies from `requirements.txt`
```
$ pip install -r requirements.txt
```

3. Install this package using
```
$ pip install -e .
```
## Usage

1. Connect the 5X board to the microcontroller. 

2. Connect the microcontroller (we recommend the Adafruit Trinket M0 or the Adafruit QT PY) to the computer using a suitable USB cable

3. Use the [Arduino IDE](https://www.arduino.cc/en/software) to upload code to a microcontroller. The code as well as upload instructions can be found in the [arduino](./arduino) folder.
If you get a `can't open device "<port-name>": Permission denied` error, modify permissions to allow read and write on that port. On Linux, this would look like 
```
$ sudo chmod a+rw <port-name>
```

4. Run test code on the computer
```
$ python tests/sensor_proc_test.py -p <port-name>
```
## Credits
This package is maintained by [Raunaq Bhirangi](https://www.cs.cmu.edu/~rbhirang/). We would also like to cite the [pyForceDAQ](https://github.com/lindemann09/pyForceDAQ) library which was used as a reference in structuring this package.
