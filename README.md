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
# Key Enhancements in This Version
## 1.Pause Function

This feature detects **[pauseX]** commands in the input text and generates corresponding silent audio signals for the specified duration. These pauses are then incorporated into the processed speech output.

### 📌 Input
- **text (str)**: The input text containing optional pause markers.
  - **Example:** `"Hello [pause2] world."`
  - Any `[pauseX]` marker represents a pause of **X seconds**.
  - X must be a positive integer (e.g., `[pause3]`, `[pause10]`).
  - The input should always be enclosed in square brackets and include the word **"pause"** followed by a numeric value.
- **sample_rate (int)**: The sampling rate used to generate the silent audio signal.

### 📌 Output
- **processed_text (str)**: The text with pause markers removed before being fed into the TTS model.
- **silent_audio (np.array)**: A NumPy array representing **X seconds of silence**.
  - **Example:** If `[pause5]` is provided, a **5-second** silent waveform is generated.
  - **Array size:** `int(X * sample_rate)`, meaning the array contains enough zero values to represent X seconds of silence.

### 📌 Function Purpose & Usage
- `pause_duration_generator(text, sample_rate)` extracts pause durations from the input text and **generates silence** for the specified time period.
- Pause commands are detected within the text using **regular expressions (regex)** and replaced with corresponding silence signals.

## 2.Error Feedback Feature

This feature evaluates the **accuracy of the Text-to-Speech (TTS) model** by comparing the **original text (TTS input) with the transcribed Speech-to-Text (STT output)**. The differences between the expected and generated text are identified and saved for analysis.

### 📌 Input
- **original_text (str)**: The input text provided to the **TTS model**.
- **audio_file_path (str)**: The **speech generated by the TTS model**, saved as an audio file.

### 📌 Output
- **stt_text (str)**: The **transcribed text from Speech-to-Text (STT)**.
- **differences (list)**: A list of detected differences between `original_text` and `stt_text`.
- **File Output**: The `save_differences()` function stores the identified differences in a file.

### 📌 Processing Steps & Function Responsibilities
1. **TTS-generated speech is converted back into text using STT:**  
   - `run_whisper_stt(audio_file_path)` processes the speech and converts it into text.  
   - **Input:** The path to the generated speech file (e.g., `"generated_audio.wav"`).  
   - **Output:** A string containing the STT transcription.

2. **Text is cleaned for better comparison:**  
   - `clean_text(text)` removes unnecessary characters and formatting inconsistencies from the transcribed text.  
   - **Input:** A string of text.  
   - **Output:** A cleaned string.

3. **The original TTS input is compared with the STT output:**  
   - `compare_texts(original_text, stt_text)` identifies discrepancies between the two texts.  
   - **Input:**  
     - `original_text`: The original text provided to the TTS model.  
     - `stt_text`: The text produced by the STT system.  
   - **Output:** A list of detected differences.

4. **Differences are saved to a file:**  
   - `save_differences(differences, output_path)` writes the detected differences to a specified file.  
   - **Input:**  
     - `differences`: A list of discrepancies.  
     - `output_path`: The file path where the differences should be saved.  
   - **Output:** The differences are stored in a text file.

### 📌 Example Usage:
```python
stt_result_text = run_whisper_stt("generated_audio.wav")  # Converts speech back to text.
differences = compare_texts("Expected text", stt_result_text)  # Compares STT output with the original TTS input.
save_differences(differences, "differences.txt")  # Saves detected differences.
```



