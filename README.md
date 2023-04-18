# Drake AI - Voice Cloning
This repository contains the code and resources needed to clone Drake's voice using a pre-trained AI model. It is based on the so-vits-svc library and uses the pre-trained model to generate synthesized speech in Drake's voice.

## Getting Started
Follow these steps to set up the environment and clone Drake's voice:

### 1. Clone the Repository
Clone this repository to your local machine.

```
git clone https://github.com/mrBlackHat777/drake-ai.git
```
### 2. Download and Extract the Pre-trained Model
Download the pre-trained model from the following link:

```
https://mega.nz/file/Sm53wAwI#4PmIrSWDrEP1-pnZb5MJpTcfoHy3OBhBOhn2FVxfyb8
```
Extract the downloaded .zip file and place the contents in the models directory.

### 3. Install Dependencies
Install the required dependencies by running:

```
pip install -r requirements.txt
```
### 4. Run the Example Script
To generate an audio file in Drake's voice, run the following command:

```
python drake_voice_cloning.py --text "Hello, this is Drake speaking."
```

Replace the text in quotes with your desired input text.

### 5. Play the Generated Audio
The synthesized audio file will be saved as output.wav in the project directory. You can listen to the generated audio using any media player.

## Using Other Voices
To use other voices, download the corresponding pre-trained models and follow the same process as described above. Replace the model files in the models directory with the new models and adjust the drake_voice_cloning.py script accordingly.

## Acknowledgments
This project is based on the so-vits-svc library and the pre-trained model was obtained from the Mega.nz link provided in the instructions. Special thanks to the contributors and developers of the original library and the pre-trained models.
