import os
import re
import sys
sys.path.append(os.path.join(sys.path[0],'text_to_speech'))
sys.path.append(os.path.join(sys.path[0],'voice_conversion'))
sys.path.append(os.path.join(sys.path[0],'lip_sync/video_retalking'))
sys.path.append(os.path.join(sys.path[0],'lip_sync/video_retalking/third_part'))
sys.path.append(os.path.join(sys.path[0],'lip_sync/video_retalking/third_part/GFPGAN'))
sys.path.append(os.path.join(sys.path[0],'lip_sync/video_retalking/third_part/GPEN'))
sys.path.append(os.path.join(sys.path[0],'lip_sync/video_retalking/checkpoints'))

import click
from pathlib import Path
# import pyttsx3
# from text_to_speech.TTS.api import TTS
import so_vits_svc_fork.__main__ as so_vits_svc_fork
from lip_sync.video_retalking import inference as lip_syn_infer
# from audio_seg import audio_slicer
from text_to_speech.bark.api import semantic_to_waveform
from text_to_speech.bark import generate_audio, SAMPLE_RATE
from text_to_speech.bark.generation import generate_text_semantic, preload_models
import librosa
import scipy
import numpy as np
import random
import nltk  # we'll use this to split into sentences


class RichHelpFormatter(click.HelpFormatter):
    def __init__(
        self,
        indent_increment: int = 2,
        width: int = None,
        max_width: int = None,
    ) -> None:
        width = 100
        super().__init__(indent_increment, width, max_width)


def patch_wrap_text():
    orig_wrap_text = click.formatting.wrap_text

    def wrap_text(
        text,
        width=78,
        initial_indent="",
        subsequent_indent="",
        preserve_paragraphs=False,
    ):
        return orig_wrap_text(
            text.replace("\n", "\n\n"),
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            preserve_paragraphs=True,
        ).replace("\n\n", "\n")

    click.formatting.wrap_text = wrap_text


patch_wrap_text()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)
click.Context.formatter_class = RichHelpFormatter

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass

@cli.command()
def init():
    os.makedirs("./data/input/text", exist_ok=True)
    os.makedirs("./data/input/video", exist_ok=True)
    os.makedirs("./data/tts", exist_ok=True)
    os.makedirs("./data/vc", exist_ok=True)
    os.makedirs("./data/lip_sync", exist_ok=True)
    os.makedirs("./data/vc_train/audio", exist_ok=True)
    os.makedirs("./data/vc_train/configs", exist_ok=True)
    os.makedirs("./data/vc_train/dataset_raw", exist_ok=True)
    os.makedirs("./data/vc_train/filelists", exist_ok=True)
    os.makedirs("./data/vc_train/model", exist_ok=True)

import whisper
import difflib



def run_whisper_stt(audio_file_path):
    model = whisper.load_model("base") 
    result = model.transcribe(audio_file_path) 
    return result["text"]

def clean_text(text):
    # It removes punctuation and changes the text to lowercase because sometimes it says Hi, and hi are different. But its the same word
    text = text.lower()
    # It removes the comma
    text = re.sub(r'[,]', '', text)
    # If it has more than one blank character, it reduces it to one.
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def compare_texts(original_text, stt_text):
    # It removes the pause commands
    original_text = re.sub(r'\[pause\d+\]', '', original_text)
    # Cleans the text
    original_text = clean_text(original_text)
    stt_text = clean_text(stt_text)
    print("Cleaned Text to speech input text: ", original_text)
    print("Cleaned Speech to text : ", stt_text)
    

    
    # It compares the texts
    differ = difflib.ndiff(original_text.split(), stt_text.split())
    return list(differ)  


def save_differences(differences, output_path):
    with open(output_path, 'w') as f:
        f.write("TTS vs STT Comparison:\n\n")
        
        tts_line = []
        stt_line = []
        
        for diff in differences:
            if diff.startswith('- '):  # TTS'de var, STT'de yok
                tts_line.append(f"**{diff[2:]}**")
            elif diff.startswith('+ '):  # STT'de var, TTS'de yok
                stt_line.append(f"**{diff[2:]}**")
            elif diff.startswith('  '):  # Aynı kelime
                tts_line.append(diff[2:])
                stt_line.append(diff[2:])
        
        # TTS ve STT çıktısını dosyaya yazdır
        f.write("TTS Output:\n")
        f.write(' '.join(tts_line) + '\n\n')
        f.write("STT Output:\n")
        f.write(' '.join(stt_line) + '\n')

def pause_duration_generator(pause_command):
    """
    pause_command: a string in the format 'pauseX', where X is the duration in seconds.
    """
    duration = int(re.findall(r'\d+', pause_command)[0])  #It extracts the number from 'pauseX'
    return np.zeros(int(duration * SAMPLE_RATE))  # It generates  pause of X seconds


##########################################
### text to speech using suno-ai bark ####
##########################################
def tts_bark_infer(input_file, speaker, output_file):
    import re

    nltk.download('punkt')
    try:
        preload_models()
    except:
        preload_models()
    GEN_TEMP = 0.6
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    
    with open(input_file, "r") as f:
        script = f.read()
    
    # Detect pause commands
    detected_commands = re.findall(r'\[pause\d+\]', script)
    
    # Replace newlines and strip the script
    script = script.replace("\n", " ").strip()

    sentences = nltk.sent_tokenize(script)

    pieces = []
    for sentence in sentences:
        # Detect and handle pause commands within sentences
        for command in detected_commands:
            if command in sentence:
                pause_duration = pause_duration_generator(command)  # Dynamic pause times
                pieces += [pause_duration]  # Adding pauses, the audio track of each sentence and the required pause times are added to the list of pieces
                sentence = sentence.replace(command, '').strip()
        
        if sentence == ".":
            pieces += [silence.copy()]
        else:
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=GEN_TEMP,
                min_eos_p=0.01,  # this controls how likely the generation is to end
            )

            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker,)
            audio_array, index = librosa.effects.trim(audio_array, top_db= 37)
            pieces += [audio_array, silence.copy()]

    scipy.io.wavfile.write(output_file, rate=SAMPLE_RATE, data=np.concatenate(pieces).astype(np.float32))

    # Run Whisper STT on the generated TTS audio, 
    stt_result_text = run_whisper_stt(output_file)

    #Defines the file name to use and writes the result to txt file.
    stt_output_file = output_file.replace(".wav", "_stt.txt") 

    with open(stt_output_file, 'w') as f:
        f.write(stt_result_text)

    # Load the original text
    with open(input_file, "r") as f:
        original_text = f.read()


    # Compare texts
    differences = compare_texts(original_text, stt_result_text)

    #if the differences have already occurred, it deletes it from the previous training and creates the file only if there is a difference.
    # On the 3rd try, there was no error and deleted the differences file.
    diff_file_path = output_file.replace(".wav", "_differences.txt")
    if os.path.exists(diff_file_path):
        os.remove(diff_file_path)

    # Save differences if any
    if differences:
        save_differences(differences, diff_file_path)
    else:
        print("No differences found between the original text and the transcribed text.")



@cli.command()
@click.option(
    "--input-path",
    type=click.Path(),
    help="path to input dir or file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="path to output dir",
)
@click.option(
    "--speaker",
    type=click.STRING,
    default="v2/de_speaker_4",
    help="bark speaker name",
)
def tts_bark(
    input_path: Path,
    output_dir: Path,
    speaker: str
):
    input_path = "./data/input/text"
    output_dir = "./data/tts"
    
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    
    if os.path.isfile(input_path):
        name = os.path.basename(input_path)
        out_path = os.path.join(output_dir, name)
        out_path = out_path + '.wav'
        tts_bark_infer(input_file=input_path, speaker=speaker, output_file=out_path)
    else:
        for name in os.listdir(input_path):
            file_path = os.path.join(input_path, name)
            if os.path.isfile(file_path):
                print("Processing ", name)
                out_path = os.path.join(output_dir, name)
                out_path = out_path + '.wav'
                tts_bark_infer(input_file=file_path, speaker=speaker, output_file=out_path)
            
##############################################
#### voice conversion using so_vits_svc_fork ####
##############################################

@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True),
    default="./data/vc_train/input_audio",
    help="path to dataset",
)
@click.option(
    "--filelist-path",
    type=click.Path(),
    default="./data/vc_train/filelists",
    help="path to file lists",
)
@click.option(
    "--config_path",
    type=click.Path(),
    default="./data/vc_train/configs/config.json",
    help="path to config.json",
)
@click.option(
    "--cache_dir",
    type=click.Path(),
    default="./data/vc_train/cache",
    help="path to temporary folder",
)
def vc_pre_processing(input_dir, filelist_path, config_path, cache_dir):
    dataset_folder = os.path.abspath("./data/vc_train/dataset_raw")
    # # clear dataset folder
    os.system(f"rm -rf {dataset_folder}/*")
    
    output_split = os.path.join(dataset_folder, "split")
    Path(output_split).mkdir(parents=True, exist_ok=True)
    # audio_slicer(input_dir=input_dir, output_dir=output_split)
    
    ## split audio into chunks
    so_vits_svc_fork.pre_split(input_dir=input_dir, output_dir=output_split)
    
    ## resample
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.abspath(cache_dir)
    os.system(f"rm -rf {cache_dir}/*")
    
    so_vits_svc_fork.pre_resample(input_dir=dataset_folder, output_dir=cache_dir)
    
    ## pre config
    filelist_path = os.path.abspath(filelist_path)
    config_path = os.path.abspath(config_path)
    so_vits_svc_fork.pre_config(input_dir=cache_dir, filelist_path = filelist_path, config_path=config_path)

    ## re-hubert
    so_vits_svc_fork.pre_hubert(input_dir=cache_dir, config_path=config_path)

@cli.command()
@click.option(
    "--config_path",
    type=click.Path(),
    default="./data/vc_train/configs/config.json",
    help="path to config.json",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default="./data/vc_train/model",
    help="path to model",
)
@click.option(
    "--max-epoch",
    type=int,
    help="max epoch",
)
def vc_train(
    config_path: Path,
    model_path: Path,
    max_epoch: int
):
    config_path = os.path.abspath(config_path)
    model_path = os.path.abspath(model_path)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    so_vits_svc_fork.train(config_path, model_path, max_epochs=max_epoch)
    

@cli.command()
@click.option(
    "--input-path",
    type=click.Path(exists=True),
    default="./data/tts",
    help="path to input dir or file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./data/vc",
    help="path to output dir",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default="./data/vc_train/model",
    help="path to model or folder",
)
@click.option(
    "--config-path",
    type=click.Path(),
    default="./data/vc_train/model/config.json",
    help="path to config.json",
)
def vc_infer(
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    config_path: Path,
):    
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path)
    
    if os.path.isfile(input_path):
        audio_name = os.path.basename(input_path)
        so_vits_svc_fork.infer(input_path= input_path,
                               output_path= os.path.join(output_dir, "converted_"+audio_name),
                               model_path= model_path,
                               config_path= config_path,)
    else:
        for audio_name in os.listdir(input_path):
            file_path= os.path.join(input_path,audio_name)
            if os.path.isfile(file_path):
                print("Processing", audio_name)
                so_vits_svc_fork.infer(input_path= file_path,
                                   output_path= os.path.join(output_dir,"converted_"+audio_name),
                                   model_path= model_path,
                                   config_path= config_path,
                                #    f0_method = "crepe"
                                   )

    

##########################################
##### lip sync using video_retalking #####
##########################################
@cli.command()
@click.option(
    "--audio-path",
    type=click.Path(exists=True),
    default="./data/vc",
    help="path to input audio dir or video file",
)
@click.option(
    "--video-path",
    type=click.Path(),
    default="./data/input/video",
    help="path to input video dir or video file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./data/lip_sync",
    help="path to output dir",
)
@click.option(
    "--start-frame",
    type=int,
    default=None,
    help="start frame",
)

def lip_sync(
    audio_path: Path,
    video_path: Path,
    output_dir: Path,
    start_frame: int
):  
    if os.path.isdir(video_path):
        files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        # randomly select a video
        video_path = os.path.join(video_path, files[random.randint(0, len(files)-1)])
    
    print("Using", os.path.basename(video_path))
    
    if os.path.isfile(audio_path):
        audio_name = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, audio_name.split(sep='.')[0])+'.mp4'
        lip_syn_infer.inference(face_path=video_path, 
                      audio_path=audio_path,
                      outfile=output_path,
                      cache_dir=os.path.dirname(video_path)+"/cache",
                      start_frame = start_frame)
    else:
        for audio_name in os.listdir(audio_path):
            print("Processing", audio_name)
            output_path = os.path.join(output_dir, audio_name.split(sep='.')[0])+'.mp4'
            lip_syn_infer.inference(face_path=video_path, 
                                    audio_path=os.path.join(audio_path,audio_name),
                                    outfile=output_path,
                                    cache_dir=os.path.dirname(video_path)+"/cache",
                                    start_frame = start_frame)
        
if __name__ == "__main__":
    cli()