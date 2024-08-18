from flask import Flask, request, jsonify
from app import create_app

app, celery = create_app()

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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File terlalu besar!"}), 413

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
