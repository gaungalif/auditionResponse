from flask import Flask, request, jsonify, abort
import librosa
import numpy as np
import speech_recognition as sr
from scipy.spatial.distance import cdist
from librosa.sequence import dtw
import io
from celery import Celery

# Inisialisasi Flask dan konfigurasi Celery
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Batas ukuran file 16 MB

# Konfigurasi Celery untuk menggunakan Redis
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Speech-to-Text menggunakan SpeechRecognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# Pemisahan kata sederhana (placeholder)
def split_text_to_words(text):
    return text.split()

# Analisis Intonasi
def extract_pitch(audio_data, sr):
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    pitch = [np.mean(pitches[magnitudes > 0], axis=0) for magnitude in magnitudes]
    return pitch

def analyze_intonation(pitch):
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)
    return pitch_mean, pitch_std

# Analisis Ritme
def detect_onsets(audio_data, sr):
    onset_times = librosa.onset.onset_detect(y=audio_data, sr=sr, units='time')
    return onset_times

def calculate_rhythm_accuracy(onset_times_input, onset_times_reference):
    onset_times_input = np.array(onset_times_input).reshape(-1, 1)
    onset_times_reference = np.array(onset_times_reference).reshape(-1, 1)
    D, _ = dtw(onset_times_input, onset_times_reference)
    average_onset_difference = np.mean(D)
    rhythm_accuracy_percentage = 100 - average_onset_difference
    return average_onset_difference, rhythm_accuracy_percentage

# Analisis Kontrol Napas
def analyze_breath_control(audio_data):
    energy = librosa.feature.rms(y=audio_data)[0]
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    return mean_energy, std_energy

# Fungsi Analisis Utama (Background Task)
@celery.task
def analyze_audio(input_audio_file, reference_audio_file):
    # Load audio files dari memory
    input_y, input_sr = librosa.load(input_audio_file, sr=None)
    reference_y, reference_sr = librosa.load(reference_audio_file, sr=None)
    
    # Transkripsi audio ke teks
    input_text = transcribe_audio(input_audio_file)
    reference_text = transcribe_audio(reference_audio_file)
    
    # Simulasi pemisahan kata
    input_words = split_text_to_words(input_text)
    reference_words = split_text_to_words(reference_text)
    
    # Analisis berdasarkan audio referensi
    onset_times_input = detect_onsets(input_y, input_sr)
    onset_times_reference = detect_onsets(reference_y, reference_sr)
    
    # Analisis Intonasi
    input_pitches = extract_pitch(input_y, input_sr)
    intonation_mean_input, intonation_std_input = analyze_intonation(input_pitches)
    
    reference_pitches = extract_pitch(reference_y, reference_sr)
    intonation_mean_reference, intonation_std_reference = analyze_intonation(reference_pitches)
    
    # Analisis Ritme
    average_onset_difference, rhythm_accuracy_percentage = calculate_rhythm_accuracy(onset_times_input, onset_times_reference)
    
    # Analisis Kontrol Napas
    mean_energy_input, std_energy_input = analyze_breath_control(input_y)
    mean_energy_reference, std_energy_reference = analyze_breath_control(reference_y)
    
    result = {
        'input_text': input_text,
        'reference_text': reference_text,
        'input_words': input_words,
        'reference_words': reference_words,
        'intonation_mean_input': intonation_mean_input,
        'intonation_std_input': intonation_std_input,
        'intonation_mean_reference': intonation_mean_reference,
        'intonation_std_reference': intonation_std_reference,
        'average_onset_difference': average_onset_difference,
        'rhythm_accuracy_percentage': rhythm_accuracy_percentage,
        'mean_energy_input': mean_energy_input,
        'std_energy_input': std_energy_input,
        'mean_energy_reference': mean_energy_reference,
        'std_energy_reference': std_energy_reference
    }
    
    return result

# Route untuk Analisis
@app.route('/analyze', methods=['POST'])
def analyze():
    input_file = request.files['input_file']
    reference_file = request.files['reference_file']
    
    input_audio_file = io.BytesIO(input_file.read())
    reference_audio_file = io.BytesIO(reference_file.read())
    
    # Jalankan analisis sebagai background task
    task = analyze_audio.delay(input_audio_file, reference_audio_file)
    
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
