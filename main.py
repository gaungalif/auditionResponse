from flask import Flask, request, jsonify
import os
import tempfile
from PIL import Image
import librosa
import numpy as np
from celery import Celery
from app.utils import (
    transcribe_audio, split_text_to_words, extract_pitch, analyze_intonation,
    calculate_rhythm_accuracy, calculate_pitch_range, calculate_speech_rate,
    calculate_intonation_accuracy, calculate_similarity_percentage, cleanse_audio,
    generate_pitch_graphs, identify_voice_type, convert_m4a_to_wav, find_false_and_unspoken_words,
    calculate_speech_rate, generate_speech_rate_graph
)


# Initialize Flask and configure Celery
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16 MB

# Initialize Celery
celery = Celery(app.name)



# Analyze audio (Background Task)
# @celery.task(name='app.main.analyze_audio')
def analyze_audio(input_audio_path, reference_audio_path):
    # Convert input and reference audio to WAV if needed
    input_audio_wav = os.path.join(tempfile.gettempdir(), 'input_audio.wav')
    reference_audio_wav = os.path.join(tempfile.gettempdir(), 'reference_audio.wav')

    # Cleanse the audio files
    cleanse_audio(input_audio_path, input_audio_wav)
    cleanse_audio(reference_audio_path, reference_audio_wav)
    
    # Transcribe audio to text
    input_text = transcribe_audio(input_audio_wav)
    reference_text = transcribe_audio(reference_audio_wav)
    
    input_words = split_text_to_words(input_text)
    reference_words = split_text_to_words(reference_text)
    
    false_words, unspoken_words = find_false_and_unspoken_words(input_text, reference_text)

    # Calculate similarity percentage
    similarity_percentage = calculate_similarity_percentage(input_words, reference_words)
    
    if similarity_percentage < 50:
        return {
            'input_words': input_words,
            'reference_words': reference_words,
            'false_words': false_words,
            'unspoken_words': unspoken_words,
            'similarity_percentage': similarity_percentage,
            'error': 'your simmilarity percentage is less than 50%, try to record again'
        }

    # Analyze intonation
    input_pitches = extract_pitch(input_audio_wav)
    reference_pitches = extract_pitch(reference_audio_wav)
    
    intonation_mean_input, intonation_std_input = analyze_intonation(input_pitches)
    intonation_mean_reference, intonation_std_reference = analyze_intonation(reference_pitches)
    
    intonation_accuracy_percentage = calculate_intonation_accuracy(intonation_mean_input, intonation_std_input, intonation_mean_reference, intonation_std_reference)
    
    # Analyze rhythm
    onset_times_input = librosa.onset.onset_detect(y=librosa.load(input_audio_wav, sr=None)[0], sr=librosa.load(input_audio_wav, sr=None)[1], units='time')
    onset_times_reference = librosa.onset.onset_detect(y=librosa.load(reference_audio_wav, sr=None)[0], sr=librosa.load(reference_audio_wav, sr=None)[1], units='time')
    
    _, rhythm_accuracy_percentage = calculate_rhythm_accuracy(onset_times_input, onset_times_reference)
    
    # Calculate speech rate
    input_speech_rate = calculate_speech_rate(input_audio_wav, input_words)
    reference_speech_rate = calculate_speech_rate(reference_audio_wav, reference_words)

    speech_rate_graph = generate_speech_rate_graph(input_speech_rate, reference_speech_rate)
    # Calculate pitch range
    input_pitch_min, input_pitch_max, _ = calculate_pitch_range(input_pitches)
    
    # Calculate median pitch
    detect_median_pitch = np.median(input_pitches) if input_pitches else 0
    
    # Generate pitch graphs
    pitch_graphs = generate_pitch_graphs(input_pitches, reference_pitches)
    
    # Identify voice type
    identified_voice_type = identify_voice_type(detect_median_pitch)

    
    return {
        'input_words': input_words,
        'reference_words': reference_words,
        'false_words': false_words,
        'unspoken_words': unspoken_words,
        'intonation_accuracy_percentage': intonation_accuracy_percentage,
        'rhythm_accuracy_percentage': rhythm_accuracy_percentage,
        'input_speech_rate': input_speech_rate,
        'reference_speech_rate': reference_speech_rate,
        'detect_median_pitch': detect_median_pitch,
        'pitch_range': (input_pitch_min, input_pitch_max),
        'identified_voice_type': identified_voice_type,
        'pitch_graphs': pitch_graphs,
        'speech_rate_graph': speech_rate_graph,


    }

# Route for analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    character_id = request.form['characterID']
    input_file = request.files['input_audio']

    # Retrieve reference audio based on characterID
    reference_audio_path = f"/path/to/reference_audios/{character_id}.wav"  # Adjust this path accordingly

    if not os.path.exists(reference_audio_path):
        return jsonify({"error": "Reference audio not found"}), 404

    # Save input m4a audio to disk
    input_m4a_path = os.path.join(tempfile.gettempdir(), 'input_audio.m4a')
    input_file.save(input_m4a_path)

    # Convert m4a to wav
    input_wav_path = os.path.join(tempfile.gettempdir(), 'input_audio.wav')
    convert_m4a_to_wav(input_m4a_path, input_wav_path)

    # Run analysis as a background task
    task = analyze_audio.delay(input_wav_path, reference_audio_path)

    return jsonify({"task_id": task.id}), 202

# Endpoint to check task status
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

# Error handler for file too large
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large!"}), 413

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)

# taskes = analyze_audio('audio/reference.wav', 'audio/reference.wav')
# print(taskes)