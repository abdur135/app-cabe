async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");
    const loading = document.getElementById("loading");
    const prediction = document.getElementById("prediction");

    if (fileInput.files.length === 0) {
        alert("Pilih gambar dulu!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.style.display = "block";
    loading.style.display = "block";
    prediction.style.display = "none";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loading.style.display = "none";
        prediction.style.display = "block";

        if (data.error) {
            prediction.innerHTML = "❌ " + data.error;
        } else {
            prediction.innerHTML = `
                <b>Hasil:</b><br><br>
                ${data.prediction}<br><br>
                <b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%
            `;
        }

    } catch (error) {
        loading.style.display = "none";
        prediction.style.display = "block";
        prediction.innerHTML = "❌ Error!";
    }
}