import os
import re
import sys
sys.path.append(os.path.join(sys.path[0], 'text_to_speech'))
sys.path.append(os.path.join(sys.path[0], 'voice_conversion'))
sys.path.append(os.path.join(sys.path[0], 'lip_sync/video_retalking'))
sys.path.append(os.path.join(sys.path[0], 'lip_sync/video_retalking/third_part'))
sys.path.append(os.path.join(sys.path[0], 'lip_sync/video_retalking/third_part/GFPGAN'))
sys.path.append(os.path.join(sys.path[0], 'lip_sync/video_retalking/third_part/GPEN'))
sys.path.append(os.path.join(sys.path[0], 'lip_sync/video_retalking/checkpoints'))

import click
from pathlib import Path
import so_vits_svc_fork.__main__ as so_vits_svc_fork
from lip_sync.video_retalking import inference as lip_syn_infer
from text_to_speech.bark.api import semantic_to_waveform
from text_to_speech.bark import generate_audio, SAMPLE_RATE
from text_to_speech.bark.generation import generate_text_semantic, preload_models
import librosa
import scipy
import numpy as np
import random
import nltk  # for sentence splitting

# -------------- New / Unified import -------------
import whisper
import difflib
# -------------------------------------------------


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


# ----------------------------------------------------------------
# 1) UNIFIED Whisper STT function
# ----------------------------------------------------------------
def run_whisper_stt(audio_file_path, timestamps=False, model_type="base"):
    """
    Unified Whisper STT function.
      - audio_file_path: path to audio file
      - timestamps     : if True, return word-level timestamps
      - model_type     : whisper model size, e.g. "base", "large"

    Returns a dictionary 'result' from Whisper:
       {
         "text": "...",
         "segments": [...]  # if timestamps=True, includes word-level info
       }
    """
    model = whisper.load_model(model_type)
    if timestamps:
        result = model.transcribe(audio_file_path, word_timestamps=True)
    else:
        result = model.transcribe(audio_file_path)
    return result


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compare_texts(original_text, stt_text):
    # Remove [pauseX] markers
    original_text = re.sub(r'\[pause\d+\]', '', original_text)
    original_text = clean_text(original_text)
    stt_text = clean_text(stt_text)
    print("Cleaned Text to speech input text:", original_text)
    print("Cleaned Speech to text:", stt_text)

    differ = difflib.ndiff(original_text.split(), stt_text.split())
    return list(differ)


def save_differences(differences, output_path):
    with open(output_path, 'w') as f:
        f.write("TTS vs STT Comparison:\n\n")

        tts_line = []
        stt_line = []

        for diff in differences:
            if diff.startswith('- '):
                tts_line.append(f"**{diff[2:]}**")
            elif diff.startswith('+ '):
                stt_line.append(f"**{diff[2:]}**")
            elif diff.startswith('  '):
                tts_line.append(diff[2:])
                stt_line.append(diff[2:])

        f.write("TTS Input:\n")
        f.write(' '.join(tts_line) + '\n\n')
        f.write("STT Output:\n")
        f.write(' '.join(stt_line) + '\n')


def pause_duration_generator(pause_command):
    """
    Extracts the integer X from a string like '[pauseX]' and returns
    a NumPy array of zeros for X seconds of silence.
    """
    duration = int(re.findall(r'\d+', pause_command)[0])  # Extract numeric value
    return np.zeros(int(duration * SAMPLE_RATE))


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
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # 0.25s silence

    with open(input_file, "r") as f:
        script = f.read()

    # Find all [pauseX] occurrences
    detected_commands = re.findall(r'\[pause\d+\]', script)

    # Remove line breaks and split into sentences
    script = script.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(script)
    pieces = []

    for sentence in sentences:
        # If sentence contains a pause command like [pause10], generate pause
        for command in detected_commands:
            if command in sentence:
                # Generate silence for the requested duration
                pieces.append(pause_duration_generator(command))
                # Remove the actual [pauseX] text from the sentence
                sentence = sentence.replace(command, '').strip()

        # If the remaining sentence is just ".", add a short silence
        if sentence == ".":
            pieces.append(silence.copy())
        else:
            # Generate semantic tokens
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=GEN_TEMP,
                min_eos_p=0.01,
            )
            # Convert semantic tokens to audio
            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=speaker)
            audio_array, index = librosa.effects.trim(audio_array, top_db=37)
            pieces += [audio_array, silence.copy()]

    # Save final concatenated audio
    final_audio = np.concatenate(pieces).astype(np.float32)
    scipy.io.wavfile.write(output_file, rate=SAMPLE_RATE, data=final_audio)

    # Run Whisper STT on generated TTS audio (using the unified function)
    stt_result = run_whisper_stt(output_file)  # default "base" model, no timestamps
    stt_text = stt_result["text"]

    # Save Whisper transcription
    stt_output_file = output_file.replace(".wav", "_stt.txt")
    with open(stt_output_file, 'w') as f:
        f.write(stt_text)

    # Compare with original text, generate diff
    with open(input_file, "r") as f:
        original_text = f.read()

    differences = compare_texts(original_text, stt_text)
    diff_file_path = output_file.replace(".wav", "_differences.txt")
    if os.path.exists(diff_file_path):
        os.remove(diff_file_path)
    if differences:
        save_differences(differences, diff_file_path)
    else:
        print("No differences found between the original text and the transcribed text.")


# ----------------------------------------------------------------
# Below: The translation command, training, voice conversion, etc.
#         (All your existing CLI commands remain unchanged)
# ----------------------------------------------------------------
from translation import translate_files

@click.group()
def cli():
    pass

@cli.command(name='translate-text')
def translate_text():
    """Translates files from German to English."""
    input_folder = "./data/input/text/german-text"
    output_folder = "./data/input/text"
    click.echo(f"Running translate_text with input: {input_folder} and output: {output_folder}")
    translate_files(input_folder, output_folder)


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
    os.system(f"rm -rf {dataset_folder}/*")
    
    output_split = os.path.join(dataset_folder, "split")
    Path(output_split).mkdir(parents=True, exist_ok=True)

    so_vits_svc_fork.pre_split(input_dir=input_dir, output_dir=output_split)
    
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.abspath(cache_dir)
    os.system(f"rm -rf {cache_dir}/*")
    
    so_vits_svc_fork.pre_resample(input_dir=dataset_folder, output_dir=cache_dir)
    filelist_path = os.path.abspath(filelist_path)
    config_path = os.path.abspath(config_path)
    so_vits_svc_fork.pre_config(input_dir=cache_dir, filelist_path=filelist_path, config_path=config_path)
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
def vc_train(config_path: Path, model_path: Path, max_epoch: int):
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
def vc_infer(input_path: Path, output_dir: Path, model_path: Path, config_path: Path):
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path)

    if os.path.isfile(input_path):
        audio_name = os.path.basename(input_path)
        so_vits_svc_fork.infer(
            input_path=input_path,
            output_path=os.path.join(output_dir, "converted_" + audio_name),
            model_path=model_path,
            config_path=config_path,
        )
    else:
        for audio_name in os.listdir(input_path):
            file_path = os.path.join(input_path, audio_name)
            if os.path.isfile(file_path):
                print("Processing", audio_name)
                so_vits_svc_fork.infer(
                    input_path=file_path,
                    output_path=os.path.join(output_dir, "converted_" + audio_name),
                    model_path=model_path,
                    config_path=config_path,
                    # f0_method="crepe"
                )


@cli.command(name='tts-bark')
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
def tts_bark_cmd(input_path: Path, output_dir: Path, speaker: str):
    input_path = "./data/input/text"
    output_dir = "./data/tts"

    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    
    if os.path.isfile(input_path):
        name = os.path.basename(input_path)
        out_path = os.path.join(output_dir, name) + '.wav'
        tts_bark_infer(input_file=input_path, speaker=speaker, output_file=out_path)
    else:
        for name in os.listdir(input_path):
            file_path = os.path.join(input_path, name)
            if os.path.isfile(file_path):
                print("Processing", name)
                out_path = os.path.join(output_dir, name) + '.wav'
                tts_bark_infer(input_file=file_path, speaker=speaker, output_file=out_path)





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
def lip_sync(audio_path: Path, video_path: Path, output_dir: Path, start_frame: int):
    if os.path.isdir(video_path):
        files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
        video_path = os.path.join(video_path, files[random.randint(0, len(files)-1)])

    print("Using", os.path.basename(video_path))

    if os.path.isfile(audio_path):
        audio_name = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, audio_name.split(sep='.')[0]) + '.mp4'
        lip_syn_infer.inference(
            face_path=video_path,
            audio_path=audio_path,
            outfile=output_path,
            cache_dir=os.path.dirname(video_path)+"/cache",
            start_frame=start_frame
        )
    else:
        for audio_name in os.listdir(audio_path):
            print("Processing", audio_name)
            output_path = os.path.join(output_dir, audio_name.split(sep='.')[0]) + '.mp4'
            lip_syn_infer.inference(
                face_path=video_path,
                audio_path=os.path.join(audio_path, audio_name),
                outfile=output_path,
                cache_dir=os.path.dirname(video_path)+"/cache",
                start_frame=start_frame
            )


if __name__ == "__main__":
    cli()
