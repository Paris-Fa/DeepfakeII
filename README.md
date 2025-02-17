# Overview
This project is the continuation of [Deepfakes_all_in_1](https://github.com/pn-pham/deepfakes_all_in_1.git), which is a user-friendly end-to-end pipline for personalized video content creation.In this work we tried to address some of the limitations found in the previous version.The pipeline utilizes user-friendly, open-source models, which include the following:
*  Text-To-Speech Model:[suno-Bark](https://github.com/suno-ai/bark.git)
*  Voice conversion :[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc.git), [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork.git)
*  Lip-sync: [video-retalking](https://github.com/OpenTalker/video-retalking.git)


# Installation

1.Install Miniconda and python
2.Create a Conda environment and install the requirements
```
git clone https://github.com/pn-pham/deepfakes_all_in_1.git

cd deepfakes_all_in_1 

conda create -n deepfakes python==3.11.5

conda activate deepfakes

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

python deepfakes.py init
```

3.Install [espeak-ng](https://pypi.org/project/espeakng/) to enable text-to-speech synthesizer.
```
pip install espeakng
```
If the installation with ```pip``` doesn't work, please use ```sudo apt-get```.
```
sudo apt-get install espeak-ng
```
4.Download [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0), then put them in ```./lip_sync/video_retalking/checkpoints```.

# Training the Voice Conversion Model
1. To train the voice conversion model, you need your own Audio dataset.The Audio should be high quality and without any background noise. the recommended length for the Audio to get an acceptable result is at least 20 minutes.

2. Place the recordings in folder ```./data/vc_train/input_audio.```
3. Run the command below to pre-process the data.
```
python deepfakes.py vc-pre-processing
```
4. Run this commant to train the voice conversion.for a small dataset eg.20 minutes it is not recommended to train the model for more than 1000 epochs.However for a larger dataset you can train it for 5000 epochs or more.The training on a Tesla V100-PCIE-32GB GPU for 5000 epochs takes about 4 days.
```
python deepfakes.py vc-train --max-epoch 5000
python deepfakes.py vc-train --max-epoch 1000
 ```
# Inference
## 1.Text-to-speech

1. prepere the script(s) you want to turn into an audio in .txt format.Each file will be used to create one audio.
2. we have added a pause function.you can add pauses with your desired duration between the sentence.while preparing the text add [pausex] .replace x with the desired duration. reemember to not add a space between pause and y.
3. place the script(s) in this folder ```./data/input/text.```.
4. Run the following command.
```
python deepfakes.py tts-bark
```

If you have several scripts but want to use only one of them use this command
```
python deepfakes.py tts-bark --input-path ./path-to-file/text.txt
```
The output audio will be store in folder ```./data/tts.```

when Bark converts text to audio there might be errors in Bark's Audio output.to find this errors user had to listen to each audio. we added an error correction feedback to automate this process,this function captures the errors of the bark output and gives you a text file that includes the comparison between the original text input and the transcription of bark audio output which we get using [open-Ai whisper](https://github.com/openai/whisper.git) .the file is stored at ```./data/tts.``` and the differences are flaged with **

## 2. Voice conversion
Bark produces and output with a defaul voice. to get the desired voice run this command to apply the voice conversion to all the audio files at ```./data/tts.``` folder.(you should first train the voice conversion model)

```
python deepfakes.py vc-infer
```
if you have several audios in your folder but want to apply voice conversion to only one of them run the following command.

```
python deepfakes.py vc-infer --input-path ./path-to-file/audio.wav

```
Note that, the latest trained model in folder ```./data/vc_train/model``` is used by default. To use a specific trained model (e.g. G_5000.pth), please include ```--model-path``` as follows:
```
python deepfakes.py vc-infer --model-path ./data/vc_train/model/G_5000.pth
```




