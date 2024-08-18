from app import celery
from app.utils import transcribe_audio, split_text_to_words, extract_pitch, analyze_intonation, detect_onsets, calculate_rhythm_accuracy, analyze_breath_control
import numpy as np
import os

@celery.task
def analyze_audio(input_audio_path, reference_audio_path):
    # Load audio files dari disk
    input_y, input_sr = librosa.load(input_audio_path, sr=None)
    reference_y, reference_sr = librosa.load(reference_audio_path, sr=None)
    
    # Transkripsi audio ke teks
    input_text = transcribe_audio(input_audio_path)
    reference_text = transcribe_audio(reference_audio_path)
    
    # Simulasi pemisahan kata
    input_words = split_text_to_words(input_text)
    reference_words = split_text_to_words(reference_text)
    
    # Analisis berdasarkan audio referensi
    onset_times_input = detect_onsets(input_audio_path)
    onset_times_reference = detect_onsets(reference_audio_path)
    
    # Analisis Intonasi
    input_pitches = extract_pitch(input_audio_path)
    intonation_mean_input, intonation_std_input = analyze_intonation(input_pitches)
    
    reference_pitches = extract_pitch(reference_audio_path)
    intonation_mean_reference, intonation_std_reference = analyze_intonation(reference_pitches)
    
    # Analisis Ritme
    average_onset_difference, rhythm_accuracy_percentage = calculate_rhythm_accuracy(onset_times_input, onset_times_reference)
    
    # Analisis Kontrol Napas
    mean_energy_input, std_energy_input = analyze_breath_control(input_audio_path)
    mean_energy_reference, std_energy_reference = analyze_breath_control(reference_audio_path)
    
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
    
    # Hapus file sementara setelah analisis selesai
    os.remove(input_audio_path)
    os.remove(reference_audio_path)
    
    return result
