# Speech Emotion Recognition using keras, tensorflow and sklearn

This repository contains the code for the speech emotion recognition model built using keras, tensorflow and sklearn libraries. The data used for trainibg purposes is the RAVDESS Audio Dataset.

Tested on python 3.11.0
### Virtual environment creation
use the following commands to create a venv -
```
python -m venv myenv
```
```
myenv/Scripts/activate
```

### Cross check the installed python version
```
python --version
```

### Installation
To install the dependencies run:
```
pip install -r requirements.txt
```

### Extracting features from audio files
To extract the data, run:
```
cd src
python ml_pipeline/utils.py
```

### Train the model
To train the model, run:
```
python ml_pipeline/model.py
```

### Use the model to make predictions
To get predictions, run:
cd src 
```
python engine.py --framework=(keras/sklearn) --infer --infer-file-path="filepath of the sample to make predictions"
```
python engine.py --framework=keras --infer --infer-file-path=../input/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-02-01-01.wav