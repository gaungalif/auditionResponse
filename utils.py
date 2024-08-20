import noisereduce as nr
from scipy.io import wavfile
import librosa
import numpy as np
import speech_recognition as sr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import io
import base64
from pydub import AudioSegment

# Function to convert m4a to wav
def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format='m4a')
    audio.export(wav_path, format='wav')

# Function to convert float32 to float for JSON serialization
def convert_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float(item) for item in obj]
    return obj

# Speech-to-Text using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# Split text into words
def split_text_to_words(text):
    return text.split()

# Extract pitch from audio
def extract_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > 0]
    
    # Filter to keep only realistic human pitch range (approximately 80 Hz to 1100 Hz)
    valid_pitches = [pitch for pitch in pitches if 80 <= pitch <= 1100]
    
    return valid_pitches

# Analyze intonation
def analyze_intonation(pitches):
    pitch_mean = np.mean(pitches) if pitches else 0
    pitch_std = np.std(pitches) if pitches else 0
    return pitch_mean, pitch_std

# Calculate rhythm accuracy
def calculate_rhythm_accuracy(onset_times_input, onset_times_reference):
    onset_times_input = np.array(onset_times_input)
    onset_times_reference = np.array(onset_times_reference)
    distance, _ = fastdtw(onset_times_input.reshape(-1, 1), onset_times_reference.reshape(-1, 1), dist=euclidean)
    average_onset_difference = np.mean(distance)
    rhythm_accuracy_percentage = 100 - average_onset_difference
    return average_onset_difference, rhythm_accuracy_percentage

# Calculate pitch range
def calculate_pitch_range(pitches):
    valid_pitches = [pitch for pitch in pitches if pitch > 0]
    
    if valid_pitches:
        pitch_min = min(valid_pitches)
        pitch_max = max(valid_pitches)
        pitch_range = pitch_max - pitch_min
    else:
        pitch_min = 0
        pitch_max = 0
        pitch_range = 0
    
    return pitch_min, pitch_max, pitch_range

# Calculate speech rate
def calculate_speech_rate(audio_path, words):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    word_count = len(words)
    speech_rate = word_count / duration
    return speech_rate

# Function to generate speech rate graph
def generate_speech_rate_graph(input_speech_rate, reference_speech_rate):
    plt.figure(figsize=(8, 6))
    
    categories = ['Input Speech Rate', 'Reference Speech Rate']
    values = [input_speech_rate, reference_speech_rate]
    
    plt.bar(categories, values, color=['blue', 'red'])
    plt.title('Speech Rate Comparison')
    plt.xlabel('Category')
    plt.ylabel('Words Per Minute (WPM)')
    
    # Save plot to image
    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plt.close()
    plot_image.seek(0)
    
    # Convert image to base64 string
    speech_rate_graph = base64.b64encode(plot_image.read()).decode('utf-8')
    return speech_rate_graph

# Calculate intonation accuracy
def calculate_intonation_accuracy(mean_input, std_input, mean_reference, std_reference):
    if mean_reference != 0:
        mean_difference = abs(mean_input - mean_reference)
        mean_accuracy_percentage = max(0, 100 - (mean_difference / mean_reference * 100))
    else:
        mean_accuracy_percentage = 0
    
    if std_reference != 0:
        std_difference = abs(std_input - std_reference)
        std_accuracy_percentage = max(0, 100 - (std_difference / std_reference * 100))
    else:
        std_accuracy_percentage = 0
    
    intonation_accuracy_percentage = (mean_accuracy_percentage + std_accuracy_percentage) / 2
    return intonation_accuracy_percentage

# Calculate similarity percentage
def calculate_similarity_percentage(input_words, reference_words):
    matching_words = set(input_words) & set(reference_words)
    total_words = len(set(input_words) | set(reference_words))
    
    if total_words == 0:
        return 0
    
    similarity_percentage = (len(matching_words) / total_words) * 100
    return similarity_percentage

# Function to perform audio cleansing
def cleanse_audio(input_audio_path, cleaned_audio_path):
    sample_rate, data = wavfile.read(input_audio_path)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    cleaned_data = nr.reduce_noise(y=data, sr=sample_rate)
    wavfile.write(cleaned_audio_path, sample_rate, cleaned_data.astype(data.dtype))

# Generate pitch graphs
def generate_pitch_graphs(input_pitches, reference_pitches):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.hist(input_pitches, bins=30, range=(80, 1100), color='blue', alpha=0.7)
    plt.title('Pitch Histogram - Input Audio')
    plt.xlabel('Pitch (Hz)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 1, 2)
    plt.hist(reference_pitches, bins=30, range=(80, 1100), color='red', alpha=0.7)
    plt.title('Pitch Histogram - Reference Audio')
    plt.xlabel('Pitch (Hz)')
    plt.ylabel('Frequency')

    # Save plot to image
    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plt.close()
    plot_image.seek(0)
    
    # Convert image to base64 string
    pitch_graphs = base64.b64encode(plot_image.read()).decode('utf-8')
    return pitch_graphs

# Identify voice type based on median pitch
def identify_voice_type(detect_median_pitch):
    voice_types = {
        "Soprano": (261.63, 1046.50),
        "Mezzo-soprano": (220.00, 880.00),
        "Contralto": (174.61, 698.46),
        "Tenor": (130.81, 523.25),
        "Baritone": (110.00, 440.00),
        "Bass": (82.41, 329.63)
    }
    
    identified_voice_type = None
    for voice, (low, high) in voice_types.items():
        if low <= detect_median_pitch <= high:
            identified_voice_type = voice
            break
    return identified_voice_type

# Function to find false and unspoken words
def find_false_and_unspoken_words(input_text, reference_text):
    # Split input and reference texts into sets of words
    input_words = set(input_text.lower().split())
    reference_words = set(reference_text.lower().split())
    
    # Identify false words: words in the input that don't match the reference
    false_words = input_words - reference_words
    
    # Identify unspoken words: words in the reference that are missing in the input
    unspoken_words = reference_words - input_words

    return list(false_words), list(unspoken_words)
