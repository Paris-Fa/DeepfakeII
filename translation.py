import requests
import os
from pathlib import Path

# Hardcoded API Key
DEEPL_API_KEY = "0aa5c1fd-b8b9-43cd-81a4-82c40397ac1a"  # Replace with your actual DeepL API key

def translate_files(input_folder, output_folder):
    """
    Translates text files from German to English using the DeepL API.
    """
    # Ensure input and output folders are absolute paths
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    # Check if output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):  # Only process text files
            input_file_path = os.path.join(input_folder, filename)
            with open(input_file_path, 'r', encoding='utf-8') as file:
                german_text = file.read()

            # Send request to DeepL API
            url = "https://api.deepl.com/v2/translate"  
            params = {
                "auth_key": DEEPL_API_KEY,
                "text": german_text,
                "source_lang": "DE",  # German source language
                "target_lang": "EN",  # English target language
            }

            try:
                response = requests.post(url, data=params)
                response.raise_for_status()  # Will raise HTTPError for bad responses (4xx or 5xx)

                # Get the translated text from the response
                translated_text = response.json()["translations"][0]["text"]

                # Write the translated text to the output folder
                output_file_path = os.path.join(output_folder, f"translated_{filename}")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(translated_text)

                print(f"Translated file saved as: {output_file_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error translating file {filename}: {str(e)}")
