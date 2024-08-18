import librosa
import numpy as np
import speech_recognition as sr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

def split_text_to_words(text):
    return text.split()

def extract_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = [np.mean(pitches[magnitudes > 0], axis=0) for magnitude in magnitudes]
    return pitch

def analyze_intonation(pitch):
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    return pitch_mean, pitch_std

def detect_onsets(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return onset_times

def calculate_rhythm_accuracy(onset_times_input, onset_times_reference):
    onset_times_input = np.array(onset_times_input)
    onset_times_reference = np.array(onset_times_reference)
    distance, _ = fastdtw(onset_times_input.reshape(-1, 1), onset_times_reference.reshape(-1, 1), dist=euclidean)
    average_onset_difference = np.mean(distance)
    rhythm_accuracy_percentage = 100 - average_onset_difference
    return average_onset_difference, rhythm_accuracy_percentage

def analyze_breath_control(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    return mean_energy, std_energy
