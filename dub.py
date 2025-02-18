import os
import re
from pydub import AudioSegment
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    vfx
)
from deepfakes import run_whisper_stt

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
output_dir                = "data/dub-output/"
final_video_output        = os.path.join(output_dir, "synced_video.mp4")
final_dubbed_video_output = os.path.join(output_dir, "final_dubbed_video.mp4")

os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------
# DYNAMIC INPUT FILE SELECTION
# ------------------------------------------------------------------
english_audio_dir = "data/vc"
german_video_dir  = "data/grinput"

def find_file_in_folder(folder_path, extension):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] The folder '{folder_path}' does not exist.")
        return None
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(f".{extension.lower()}"):
            return os.path.join(folder_path, filename)
    return None

english_audio_path = find_file_in_folder(english_audio_dir, "wav")
if not english_audio_path:
    print(f"[ERROR] No .wav file found in {english_audio_dir}. Exiting.")
    exit(1)

german_video_path = find_file_in_folder(german_video_dir, "mp4")
if not german_video_path:
    print(f"[ERROR] No .mp4 file found in {german_video_dir}. Exiting.")
    exit(1)

print("English audio found at:", english_audio_path)
print("German video found at:", german_video_path)

# ------------------------------------------------------------------
# NEW: Import the unified Whisper STT from deepfakes.py
# ------------------------------------------------------------------
from deepfakes import run_whisper_stt

# ------------------------------------------------------------------
# STEP 0: EXTRACT GERMAN AUDIO FROM VIDEO
# ------------------------------------------------------------------
def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    clip.close()
    return output_audio_path

# ------------------------------------------------------------------
# Transcribe audio using run_whisper_stt
# ------------------------------------------------------------------
def transcribe_audio(input_file, model_type="large"):
    """
    Transcribes the input_file using run_whisper_stt from deepfakes.py
    with word-level timestamps. Returns (segments, full_text).
    """
    # Convert input_file to mono just in case
    temp_mono_path = os.path.join(output_dir, "temp_audio_mono.wav")
    audio = AudioSegment.from_file(input_file).set_channels(1)
    audio.export(temp_mono_path, format="wav")

    # Call the unified function with timestamps=True
    # so we get segments that contain word-level times.
    result = run_whisper_stt(temp_mono_path, timestamps=True, model_type=model_type)
    segments = result.get("segments", [])
    full_text = result.get("text", "")
    return segments, full_text

# ------------------------------------------------------------------
# STEP 2: DETECT SENTENCES AND TIMESTAMPS
# ------------------------------------------------------------------
def detect_sentences_and_timestamps(transcription_segments):
    """
    Group Whisper's segments into sentences based on punctuation.
    Returns a list of dicts:
      { "sentence", "start_time", "end_time" }
    """
    sentence_endings = re.compile(r"[.!?]")
    sentences = []
    current_sentence = ""
    current_start_time = None

    for seg in transcription_segments:
        text = seg['text'].strip()
        start_time = seg['start']
        end_time   = seg['end']

        if not text:
            continue

        if current_start_time is None:
            current_start_time = start_time

        current_sentence += (" " + text) if current_sentence else text

        # If the last character is punctuation => end of sentence
        if sentence_endings.search(text[-1:]):
            sentences.append({
                "sentence": current_sentence.strip(),
                "start_time": current_start_time,
                "end_time": end_time
            })
            current_sentence = ""
            current_start_time = None

    # leftover
    if current_sentence.strip():
        sentences.append({
            "sentence": current_sentence.strip(),
            "start_time": current_start_time,
            "end_time": transcription_segments[-1]["end"]
        })

    return sentences

# ------------------------------------------------------------------
# STEP 3: ALIGN SENTENCES
# ------------------------------------------------------------------
def align_sentences(english_sentences, german_sentences):
    aligned = []
    for eng, ger in zip(english_sentences, german_sentences):
        aligned.append({
            "english_start": eng["start_time"],
            "english_end":   eng["end_time"],
            "english_sent":  eng["sentence"],

            "german_start":  ger["start_time"],
            "german_end":    ger["end_time"],
            "german_sent":   ger["sentence"],
        })
    return aligned

# ------------------------------------------------------------------
# STEP 4: BUILD SEGMENTS (INCLUDING GAPS)
# ------------------------------------------------------------------
def build_segments_including_gaps(aligned_sentences, video_duration):
    """
    Build segments for speed-matching German video to English timing, including
    silence/gaps between sentences. The final segment is extended to the end
    of the German video but is time-compressed/stretched to end exactly at
    the last English sentence end-time.
    """
    segments = []
    n = len(aligned_sentences)
    if n == 0:
        return segments

    # 1) Build segments for everything except the last one
    for i in range(n - 1):
        e_start_i    = aligned_sentences[i]["english_start"]
        e_start_next = aligned_sentences[i+1]["english_start"]

        g_start_i    = aligned_sentences[i]["german_start"]
        g_start_next = aligned_sentences[i+1]["german_start"]

        if g_start_next > video_duration:
            g_start_next = video_duration

        chunk_ger_dur = g_start_next - g_start_i
        chunk_eng_dur = e_start_next - e_start_i

        if chunk_ger_dur > 0 and chunk_eng_dur > 0:
            segments.append({
                "start":      g_start_i,
                "end":        g_start_next,
                "target_dur": chunk_eng_dur
            })

    # 2) Handle the *last* sentence differently
    last = aligned_sentences[-1]
    g_start_last = last["german_start"]
    e_start_last = last["english_start"]
    e_end_last   = last["english_end"]

    # We'll include *all* remaining German video from g_start_last -> video_duration
    # and compress/expand it to fit the final English sentence duration.
    final_ger_start = g_start_last
    final_ger_end   = video_duration+1  # entire tail of the German video
    final_ger_dur   = final_ger_end - final_ger_start+1
    final_eng_dur   = e_end_last - e_start_last+1

    if final_ger_dur > 0 and final_eng_dur > 0:
        segments.append({
            "start":      final_ger_start,
            "end":        final_ger_end,
            "target_dur": final_eng_dur
        })

    return segments

# ------------------------------------------------------------------
# STEP 5: UNIFORM SPEED SUBCLIP
# ------------------------------------------------------------------
def uniform_speed_subclip(clip, target_duration):
    original_duration = clip.duration
    if original_duration <= 0:
        return clip
    factor = original_duration / target_duration
    return clip.fx(vfx.speedx, factor)

def cut_speedmatch_video_including_gaps(video_path, segments, output_file):
    video = VideoFileClip(video_path)
    clips = []

    for seg in segments:
        seg_start  = seg["start"]
        seg_end    = seg["end"]
        target_dur = seg["target_dur"]

        if seg_start >= seg_end:
            continue
        if seg_end > video.duration:
            seg_end = video.duration

        subclip = video.subclip(seg_start, seg_end)
        subclip_var = uniform_speed_subclip(subclip, target_dur)
        clips.append(subclip_var)

    if not clips:
        print("[WARN] No valid subclips - result will be empty.")
        video.close()
        return None

    final_video = concatenate_videoclips(clips, method="chain")
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")
    video.close()
    return final_video

# ------------------------------------------------------------------
# STEP 6: COMBINE VIDEO WITH ENGLISH AUDIO
# ------------------------------------------------------------------
def combine_video_and_english_audio(video_path, english_audio_path, output_path):
    final_video   = VideoFileClip(video_path)
    english_audio = AudioFileClip(english_audio_path)

    final_video = final_video.set_audio(english_audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    final_video.close()
    english_audio.close()
    print("[INFO] Final dubbed video saved:", output_path)

# ------------------------------------------------------------------
# MAIN WORKFLOW
# ------------------------------------------------------------------
if __name__ == "__main__":
    # STEP 0) Extract German audio
    print("[INFO] Extracting German audio...")
    extracted_german_audio_path = os.path.join(output_dir, "germanaudio_extracted.wav")
    extract_audio_from_video(german_video_path, extracted_german_audio_path)
    print("[INFO] German audio extraction done.")

    # STEP 1) Transcribe English & German
    print("[INFO] Transcribing English audio...")
    try:
        eng_segments, _ = transcribe_audio(english_audio_path, model_type="small") 
        print("[INFO] English transcription complete.")
    except Exception as e:
        print("[ERROR] English transcription failed:", e)
        exit(1)

    print("[INFO] Transcribing German audio...")
    try:
        ger_segments, _ = transcribe_audio(extracted_german_audio_path, model_type="small")
        print("[INFO] German transcription complete.")
    except Exception as e:
        print("[ERROR] German transcription failed:", e)
        exit(1)

    # STEP 2) Detect sentences
    english_sents = detect_sentences_and_timestamps(eng_segments)
    german_sents  = detect_sentences_and_timestamps(ger_segments)
    print(f"Found {len(english_sents)} English sentences, {len(german_sents)} German sentences.")

    # STEP 3) Align
    aligned = align_sentences(english_sents, german_sents)

    # STEP 4) Build segments
    german_full = VideoFileClip(german_video_path)
    segments = build_segments_including_gaps(aligned, german_full.duration)
    german_full.close()

    # STEP 5) Speed-match
    speedmatched_video = cut_speedmatch_video_including_gaps(
        video_path=german_video_path,
        segments=segments,
        output_file=final_video_output
    )

    if speedmatched_video:
        speedmatched_video.close()

        # STEP 6) Overlay the original English audio
        combine_video_and_english_audio(
            final_video_output,
            english_audio_path,
            final_dubbed_video_output
        )
        print("\nAll done! The final result is in:", final_dubbed_video_output)
    else:
        print("[ERROR] Could not create a final speed-matched video.")
