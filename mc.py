from flask import Flask, request, jsonify
import librosa
import numpy as np
import speech_recognition as sr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import tempfile
from celery import Celery

# Inisialisasi Flask dan konfigurasi Celery
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Batas ukuran file 16 MB

# # Konfigurasi Celery
# app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery = Celery(app.name)
# celery.conf.update(app.config)

# Fungsi untuk mengonversi float32 menjadi float agar bisa diserialisasi ke JSON
def convert_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float(item) for item in obj]
    return obj

# Speech-to-Text menggunakan SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# Pemisahan kata sederhana (placeholder)
def split_text_to_words(text):
    return text.split()

# Analisis Intonasi
# def extract_pitch(audio_path):
#     y, sr = librosa.load(audio_path, sr=None)
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#     pitch = [np.mean(pitches[magnitudes > 0], axis=0) for magnitude in magnitudes]
#     return pitch

def extract_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > 0]  # Use only pitches where magnitude is non-zero
    
    # Filter to keep only realistic human pitch range (approximately 64 Hz to 3000 Hz)
    valid_pitches = [pitch for pitch in pitches if 64 <= pitch <= 3000]
    
    return valid_pitches

def analyze_intonation(pitch):
    pitch_mean = np.mean(pitch) if pitch else 0  # Menghindari error jika pitch kosong
    pitch_std = np.std(pitch) if pitch else 0
    return pitch_mean, pitch_std


# Analisis Ritme
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

# Analisis Kontrol Napas
def analyze_breath_control(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    return mean_energy, std_energy

# Menghitung Pitch Range
def calculate_pitch_range(pitches):
    # Hanya gunakan pitch yang valid (non-zero)
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


# Menghitung Speech Rate
def calculate_speech_rate(audio_path, words):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    word_count = len(words)
    speech_rate = word_count / duration
    return speech_rate

def calculate_intonation_accuracy(mean_input, std_input, mean_reference, std_reference):
    # Pastikan mean_reference tidak nol untuk menghindari pembagian dengan nol
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
    
    # Menggabungkan akurasi mean dan std untuk menghitung akurasi intonasi keseluruhan
    intonation_accuracy_percentage = (mean_accuracy_percentage + std_accuracy_percentage) / 2
    
    return intonation_accuracy_percentage

def calculate_similarity_percentage(input_words, reference_words):
    matching_words = set(input_words) & set(reference_words)
    total_words = len(set(input_words) | set(reference_words))
    
    if total_words == 0:
        return 0
    
    similarity_percentage = (len(matching_words) / total_words) * 100
    return similarity_percentage

# Fungsi Analisis Utama (Background Task)
# @celery.task(name='app.main.analyze_audio')
def analyze_audio(input_audio_path, reference_audio_path):
    # Load audio files dari disk
    # input_y, input_sr = librosa.load(input_audio_path, sr=None)
    # reference_y, reference_sr = librosa.load(reference_audio_path, sr=None)
    
    # Transkripsi audio ke teks
    input_text = transcribe_audio(input_audio_path)
    reference_text = transcribe_audio(reference_audio_path)
    
    # Simulasi pemisahan kata
    input_words = split_text_to_words(input_text)
    reference_words = split_text_to_words(reference_text)
    
    similarity_percentage = calculate_similarity_percentage(input_words, reference_words)
    
    if similarity_percentage < 50:
        return f"Audio berbeda, persentase kesesuaianmu = {similarity_percentage}, input words = {input_words}, reference words = {reference_words}"

    # Analisis berdasarkan audio referensi
    onset_times_input = detect_onsets(input_audio_path)
    onset_times_reference = detect_onsets(reference_audio_path)
    
    # Analisis Intonasi
    input_pitches = extract_pitch(input_audio_path)
    intonation_mean_input, intonation_std_input = analyze_intonation(input_pitches)
    
    reference_pitches = extract_pitch(reference_audio_path)
    intonation_mean_reference, intonation_std_reference = analyze_intonation(reference_pitches)
    
    # Menghitung Pitch Range
    input_pitch_min, input_pitch_max, input_pitch_range = calculate_pitch_range(input_pitches)
    # reference_pitch_min, reference_pitch_max, reference_pitch_range = calculate_pitch_range(reference_pitches)
    
    # Menghitung Akurasi Intonasi
    intonation_accuracy_percentage = calculate_intonation_accuracy(intonation_mean_input, intonation_std_input, intonation_mean_reference, intonation_std_reference)
    
    # Analisis Ritme
    average_onset_difference, rhythm_accuracy_percentage = calculate_rhythm_accuracy(onset_times_input, onset_times_reference)
    
    # Analisis Kontrol Napas
    # mean_energy_input, std_energy_input = analyze_breath_control(input_audio_path)
    # mean_energy_reference, std_energy_reference = analyze_breath_control(reference_audio_path)
    
    # Menghitung Speech Rate
    input_speech_rate = calculate_speech_rate(input_audio_path, input_words)
    # reference_speech_rate = calculate_speech_rate(reference_audio_path, reference_words)
    
    result = {
        # 'input_text': input_text,
        # 'reference_text': reference_text,
        'input_words': input_words,
        'reference_words': reference_words,
        # 'intonation_mean_input': intonation_mean_input,
        # 'intonation_std_input': intonation_std_input,
        # 'intonation_mean_reference': intonation_mean_reference,
        # 'intonation_std_reference': intonation_std_reference,
        'input_pitch_min': input_pitch_min,
        'input_pitch_max': input_pitch_max,
        'input_pitch_range': input_pitch_range,
        # 'reference_pitch_min': reference_pitch_min,
        # 'reference_pitch_max': reference_pitch_max,
        # 'reference_pitch_range': reference_pitch_range,
        'intonation_accuracy_percentage': intonation_accuracy_percentage,
        # 'average_onset_difference': average_onset_difference,
        'rhythm_accuracy_percentage': rhythm_accuracy_percentage,
        # 'mean_energy_input': mean_energy_input,
        # 'std_energy_input': std_energy_input,
        # 'mean_energy_reference': mean_energy_reference,
        # 'std_energy_reference': std_energy_reference,
        'input_speech_rate': input_speech_rate,
        # 'reference_speech_rate': reference_speech_rate
    }

    # Konversi hasil ke float agar bisa diserialisasi ke JSON
    result = convert_to_float(result)
    
    # Hapus file sementara setelah analisis selesai
    os.remove(input_audio_path)
    os.remove(reference_audio_path)
    
    return result


# Route untuk Analisis
@app.route('/analyze', methods=['POST'])
def analyze():
    input_file = request.files['input_file']
    reference_file = request.files['reference_file']
    
    # Simpan file audio ke disk
    input_audio_path = os.path.join(tempfile.gettempdir(), 'input_audio.wav')
    reference_audio_path = os.path.join(tempfile.gettempdir(), 'reference_audio.wav')
    
    input_file.save(input_audio_path)
    reference_file.save(reference_audio_path)
    
    # Jalankan analisis sebagai background task
    task = analyze_audio.delay(input_audio_path, reference_audio_path)
    
    return jsonify({"task_id": task.id}), 202

# Endpoint untuk mengecek status task
@app.route('/status/<task_id>')
def task_status(task_id):
    task = analyze_audio.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is still in progress...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # exception raised
        }
    return jsonify(response)

# Error Handler untuk File Terlalu Besar
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File terlalu besar!"}), 413

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", debug=True, port=5000)

task = analyze_audio('audio/input.wav', 'audio/reference.wav')