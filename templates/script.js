document.getElementById('file-upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = `Error: ${data.error}`;
        } else {
            document.getElementById('result').innerText = `Predicted Rice Type: ${data.predicted_class}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = `Error: ${error.message}`;
    });
});
