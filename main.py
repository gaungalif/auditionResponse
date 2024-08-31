from flask import Flask, request, jsonify
import os
import tempfile
from PIL import Image
import librosa
import numpy as np
from celery import Celery

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
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16 MB

# Initialize Celery
celery = Celery(app.name)
# Function to convert m4a to wav
def convert_m4a_to_wav(m4a_path, wav_path):
    audio = AudioSegment.from_file(m4a_path, format='m4a')
    audio.export(wav_path, format='wav')

# Function to convert float32 to float for JSON serialization
def convert_to_float(obj):
    # print(f"Converting: {obj} ({type(obj)})")
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
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    duration_minutes = duration_seconds / 60
    word_count = len(words)
    speech_rate_wpm = word_count / duration_minutes
    
    return speech_rate_wpm


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


def identify_speech_rate(speech_rate_wpm):
    speech_rate_categories = {
        "Leisure Pace": {
            "range": (0, 110),
            "feedback": "Your speech is at a Leisure Pace, giving your words time to resonate. It’s great for thoughtful moments but might need a slight speed boost in conversations.",
            "tips": "Practice slightly faster speech to keep listeners engaged while maintaining clarity."
        },
        "Conversational Flow": {
            "range": (120, 160),
            "feedback": "You're at a Conversational Flow—natural and engaging. This pace works well for everyday interactions.",
            "tips": "Vary your speed slightly to emphasize key points or add excitement."
        },
        "Expressive Pace": {
            "range": (160, 200),
            "feedback": "Your Expressive Pace adds energy and excitement. Just be careful to maintain clarity at this speed.",
            "tips": "Focus on clear articulation to ensure your words remain understandable."
        },
        "Narrator's Tempo": {
            "range": (150, 160),
            "feedback": "You're at a Narrator's Tempo, ideal for clear, thoughtful delivery. This pace works well for storytelling.",
            "tips": "Vary your speed slightly to highlight important moments."
        },
        "Speed Talk": {
            "range": (250, 400),
            "feedback": "Your Speed Talk is impressively fast, perfect for rapid delivery. Just watch for clarity.",
            "tips": "Practice enunciating key words even at high speeds to keep your message clear."
        }
    }

    identified_speech_rate_category = None
    speech_rate_feedback = {}
    
    for category, attributes in speech_rate_categories.items():
        low, high = attributes["range"]
        if low <= speech_rate_wpm <= high:
            identified_speech_rate_category = category
            speech_rate_feedback = {
                "name": "speech_rate",
                "value": f"{int(speech_rate_wpm)} wpm",
                "type": identified_speech_rate_category,
                "feedback": attributes["feedback"],
                "tips": attributes["tips"]
            }
            break
    
    return speech_rate_feedback

# Identify voice type based on median pitch
def identify_voice_type(detect_median_pitch):
    voice_types = {
        "Mezzo-Soprano": {
            "range": (220.00, 700.00),
            "detail": "Your pitch range aligns with the Mezzo-Soprano category, typically between A3 and A5. This voice type balances warmth and brightness, allowing for a rich, expressive tone.",
            "tips": "Keep developing your middle register, focusing on smooth transitions between lower and higher notes."
        },
        "Contralto": {
            "range": (175.00, 600.00),
            "detail": "Your pitch range is within the Contralto category, typically between F3 and F5. As the lowest female voice, your range has a deep, rich quality that brings warmth and depth to your singing.",
            "tips": "Continue to explore the full potential of your lower register while maintaining clarity and strength."
        },
        "Tenor": {
            "range": (130.00, 520.00),
            "detail": "Your pitch range fits within the Tenor category, typically between C3 and C5. This is the highest male voice type, known for its bright, ringing tone.",
            "tips": "Focus on refining your upper range while maintaining control over your higher notes."
        },
        "Baritone": {
            "range": (110.00, 440.00),
            "detail": "Your pitch range falls within the Baritone category, typically between A2 and A4. As a middle male voice, your range combines depth and warmth with the ability to reach higher notes.",
            "tips": "Keep practicing to ensure smooth transitions across your range and maintain a strong, resonant tone."
        },
        "Bass": {
            "range": (82.00, 330.00),
            "detail": "Your pitch range aligns with the Bass category, typically between E2 and E4. This is the lowest male voice type, characterized by deep, resonant tones.",
            "tips": "Focus on strengthening your lower register while ensuring clarity and power in your low notes. Your voice brings a strong foundation to any performance—embrace its depth and authority."
        }
    }

    identified_voice_type = None
    pitch_range_detail = {}
    
    for voice, attributes in voice_types.items():
        low, high = attributes["range"]
        if low <= detect_median_pitch <= high:
            identified_voice_type = voice
            pitch_range_detail = {
                "name": "pitch_range",
                "value": f"{low} - {high} Hz",
                "type": identified_voice_type,
                "detail": attributes["detail"],
                "tips": attributes["tips"]
            }
            break
    
    return pitch_range_detail

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



# Analyze audio (Background Task)
@celery.task(name='app.main.analyze_audio')
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
        result = {
            # 'input_words': input_words,
            # 'reference_words': reference_words,
            # 'false_words': false_words,
            # 'unspoken_words': unspoken_words,
            # 'similarity_percentage': similarity_percentage,
            'error': 'your similarity percentage is less than 50%, try to record again'
        }
        return convert_to_float(result)

    # Analyze intonation
    input_pitches = extract_pitch(input_audio_wav)
    reference_pitches = extract_pitch(reference_audio_wav)
    
    intonation_mean_input, intonation_std_input = convert_to_float(analyze_intonation(input_pitches))
    intonation_mean_reference, intonation_std_reference = convert_to_float(analyze_intonation(reference_pitches))
    
    intonation_accuracy_percentage = convert_to_float(calculate_intonation_accuracy(intonation_mean_input, intonation_std_input, intonation_mean_reference, intonation_std_reference))
    
    # Analyze rhythm
    onset_times_input = librosa.onset.onset_detect(y=librosa.load(input_audio_wav, sr=None)[0], sr=librosa.load(input_audio_wav, sr=None)[1], units='time')
    onset_times_reference = librosa.onset.onset_detect(y=librosa.load(reference_audio_wav, sr=None)[0], sr=librosa.load(reference_audio_wav, sr=None)[1], units='time')
    
    _, rhythm_accuracy_percentage = convert_to_float(calculate_rhythm_accuracy(onset_times_input, onset_times_reference))
    
    # Calculate speech rate
    input_speech_rate = calculate_speech_rate(input_audio_wav, input_words)
    identified_speech_rate = identify_speech_rate(input_speech_rate)
    reference_speech_rate = calculate_speech_rate(reference_audio_wav, reference_words)

    speech_rate_graph = generate_speech_rate_graph(input_speech_rate, reference_speech_rate)
    # Calculate pitch range
    input_pitch_min, input_pitch_max, _ = calculate_pitch_range(input_pitches)
    
    # Calculate median pitch
    detect_median_pitch = convert_to_float(np.median(input_pitches) if input_pitches else 0)
    
    # Generate pitch graphs
    pitch_graphs = generate_pitch_graphs(input_pitches, reference_pitches)
    
    # Identify voice type
    identified_voice_type = identify_voice_type(detect_median_pitch)
    
    # Compile the result
    # result = {
    #     'input_words': input_words,
    #     'reference_words': reference_words,
    #     'false_words': false_words,
    #     'unspoken_words': unspoken_words,
    #     'intonation_accuracy_percentage': str(intonation_accuracy_percentage),
    #     'rhythm_accuracy_percentage': str(rhythm_accuracy_percentage),
    #     'input_speech_rate': str(input_speech_rate),
    #     'reference_speech_rate': str(reference_speech_rate),
    #     'detect_median_pitch': str(detect_median_pitch),
    #     'pitch_range': str((input_pitch_min, input_pitch_max)),
    #     'identified_voice_type': str(identified_voice_type),
    #     'pitch_graphs': pitch_graphs,
    #     'speech_rate_graph': speech_rate_graph,
    # }
    # Convert float32 to standard Python floats before returning the result
    # res_converted =  convert_to_float(result)

    # print(type(res_converted))
    # print(type(result))
    # return type of result
    intonation = {
        "name": "intonation",
        "accuracy": str(int(intonation_accuracy_percentage)),   
    }
    rhythm = {
        "name": "rhythm",
        "accuracy": str(int(rhythm_accuracy_percentage)),
    }



    return {
                "AuditionResult":[
                    intonation,
                    rhythm,   
                ],
                "VocalProfile":[
                    identified_voice_type,
                    identified_speech_rate,
                ]
            }


# Route for analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    print(request)
    print(request.form)
    character_id = request.form['characterID']
    print('masuk2')
    input_file = request.files['input_audio']
    # Retrieve reference audio based on characterID
    print('masuk3')
    reference_audio_path = f"./audio/{character_id}.wav"  # Adjust this path accordingly

    print('masuk4')
    if not os.path.exists(reference_audio_path):
        print('masuk5')
        return jsonify({"error": "Reference audio not found"}), 404

    print('masuk6')
    # Save input m4a audio to disk
    input_m4a_path = os.path.join(tempfile.gettempdir(), 'input_audio.m4a')
    print('masuk7')
    input_file.save(input_m4a_path)

    # Convert m4a to wav
    print('masuk8')
    input_wav_path = os.path.join(tempfile.gettempdir(), 'input_audio.wav')
    print('masuk9')
    convert_m4a_to_wav(input_m4a_path, input_wav_path)

    print('masuk10')
    # Run analysis as a background task
    task = analyze_audio.delay(input_wav_path, reference_audio_path)
    print('masuk11')

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