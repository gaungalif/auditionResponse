from flask import Flask, request, jsonify
import librosa
import numpy as np
import speech_recognition as sr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# app = Flask(__name__)

# Speech-to-Text using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# Placeholder for a simple word separation
def split_text_to_words(text):
    return text.split()

# Intonation Analysis
def extract_pitch(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = [np.mean(pitches[magnitudes > 0], axis=0) for magnitude in magnitudes]
    return pitch

def analyze_intonation(pitch):
    pitch = np.array(pitch, dtype=np.float64)  # Ensure it's float64 for JSON compatibility
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    return pitch_mean, pitch_std

# Rhythm Analysis
def detect_onsets(audio_path):
    y, sr = librosa.load(audio_path)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return onset_times

def calculate_rhythm_accuracy(onset_times_input, onset_times_reference):
    try:
        if len(onset_times_input) == 0 or len(onset_times_reference) == 0:
            return None, None
        
        onset_times_input = np.array(onset_times_input, dtype=np.float64).reshape(-1, 1)
        onset_times_reference = np.array(onset_times_reference, dtype=np.float64).reshape(-1, 1)
        
        # Ensure that the arrays are in the correct shape for DTW
        if onset_times_input.shape[1] != 1 or onset_times_reference.shape[1] != 1:
            raise ValueError("Onset times should have shape (-1, 1).")
        
        # Perform DTW using fastdtw
        distance, path = fastdtw(onset_times_input, onset_times_reference, dist=euclidean)
        
        # Convert the output to float for JSON serialization
        average_onset_difference = float(distance) if distance is not None else None
        rhythm_accuracy_percentage = 100 - average_onset_difference if average_onset_difference is not None else None
        
        return average_onset_difference, rhythm_accuracy_percentage
    
    except Exception as e:
        print(f"Error in rhythm calculation: {e}")
        return None, None

# Breath Control Analysis
def analyze_breath_control(audio_path):
    y, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=y)[0]
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    return mean_energy, std_energy

# Function to calculate similarity percentage
def calculate_similarity_percentage(input_words, reference_words):
    matching_words = set(input_words) & set(reference_words)
    total_words = len(set(input_words) | set(reference_words))
    
    if total_words == 0:
        return 0
    
    similarity_percentage = (len(matching_words) / total_words) * 100
    return similarity_percentage

# Main Analysis Function
def analyze_audio(input_audio_path, reference_audio_path, similarity_threshold=50):
    # Perform speech-to-text transcription
    input_text = transcribe_audio(input_audio_path)
    reference_text = transcribe_audio(reference_audio_path)
    
    # Simulate word splitting (this is a placeholder, you might need a more advanced method)
    input_words = split_text_to_words(input_text)
    reference_words = split_text_to_words(reference_text)
    
    # Calculate similarity
    similarity_percentage = calculate_similarity_percentage(input_words, reference_words)
    
    if similarity_percentage < similarity_threshold:
        return {'message': 'Audio berbeda'}
    
    # Analysis using the reference audio
    onset_times_input = detect_onsets(input_audio_path)
    onset_times_reference = detect_onsets(reference_audio_path)
    
    # Intonation analysis
    input_pitches = extract_pitch(input_audio_path)
    intonation_mean_input, intonation_std_input = analyze_intonation(input_pitches)
    
    reference_pitches = extract_pitch(reference_audio_path)
    intonation_mean_reference, intonation_std_reference = analyze_intonation(reference_pitches)
    
    # Rhythm analysis
    average_onset_difference, rhythm_accuracy_percentage = calculate_rhythm_accuracy(onset_times_input, onset_times_reference)
    
    # Breath control analysis
    mean_energy_input, std_energy_input = analyze_breath_control(input_audio_path)
    mean_energy_reference, std_energy_reference = analyze_breath_control(reference_audio_path)
    
    result = {
        'input_text': input_text,
        'reference_text': reference_text,
        'input_words': input_words,
        'reference_words': reference_words,
        'intonation_mean_input': float(intonation_mean_input),  # Convert to float
        'intonation_std_input': float(intonation_std_input),    # Convert to float
        'intonation_mean_reference': float(intonation_mean_reference),  # Convert to float
        'intonation_std_reference': float(intonation_std_reference),    # Convert to float
        'average_onset_difference': average_onset_difference,
        'rhythm_accuracy_percentage': rhythm_accuracy_percentage,
        'mean_energy_input': float(mean_energy_input),  # Convert to float
        'std_energy_input': float(std_energy_input),    # Convert to float
        'mean_energy_reference': float(mean_energy_reference),  # Convert to float
        'std_energy_reference': float(std_energy_reference),    # Convert to float
        'similarity_percentage': similarity_percentage
    }
    
    return result

@app.route('/analyze', methods=['POST'])
def analyze():
    input_file = request.files['input_file']
    reference_file = request.files['reference_file']
    
    input_file.save(input_audio_path)
    reference_file.save(reference_audio_path)
    
    result = analyze_audio(input_audio_path, reference_audio_path)
    
    # Clean up the uploaded files
    os.remove(input_audio_path)
    os.remove(reference_audio_path)
    
    return jsonify(result)

if __name__ == '__main__':
    # app.run(debug=True)
    res = analyze()
    print(res)
