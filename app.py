from flask import Flask, render_template, request
import os
import joblib
import numpy as np
from feature_extraction import extract_features, preprocess_resize, convert_to_hsv, convert_to_grayscale
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB batas ukuran file
app.config['MAX_CONTENT_PATH'] = 200 * 1024 * 1024  # 200MB batas total path

# Load model dan label encoder
model = joblib.load('p_random_forest_model_ratio_0.80_best.pkl')
label_encoder = joblib.load('label_encoder_ratio_0.80.pkl')

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template(
        'error.html', 
        error_message="Ukuran file terlalu besar (maksimal 50MB)"
    ), 413

# Fungsi untuk memeriksa ekstensi file yang diizinkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah", 400

    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong", 400

    # Simpan file yang diunggah
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess gambar
    try:
        original_image, preprocessed_image = preprocess_resize(file_path)
        if original_image is None or preprocessed_image is None:
            return "Gagal memproses gambar", 400

        # Simpan gambar yang dipreprocess
        preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_' + file.filename)
        cv2.imwrite(preprocessed_path, preprocessed_image)

        # Konversi ke HSV
        hsv_image = convert_to_hsv(preprocessed_image)
        hsv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hsv_' + file.filename)
        cv2.imwrite(hsv_path, hsv_image)

        # Konversi ke Grayscale
        grayscale_image = convert_to_grayscale(preprocessed_image)
        grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale_' + file.filename)
        cv2.imwrite(grayscale_path, grayscale_image)

        # Ekstraksi fitur
        features = extract_features(file_path)
        if features is None:
            return "Gagal mengekstraksi fitur", 400
    except Exception as e:
        return f"Error saat memproses gambar: {str(e)}", 400

    # Prediksi probabilitas
    try:
        probabilities = model.predict_proba([features])[0]
        predicted_label = label_encoder.inverse_transform([np.argmax(probabilities)])[0]

        # Format probabilitas ke persentase
        class_probabilities = {
            label: f"{prob * 100:.1f}%" for label, prob in zip(label_encoder.classes_, probabilities)
        }
    except Exception as e:
        return f"Error saat prediksi: {str(e)}", 400

    # Informasi penyakit
    disease_info = {
        "Bercak Daun": {
            "nama_latin": "Bipolaris maydis / Helminthosporium maydis",
            "penyebab": [
                "Disebabkan oleh cendawan Bipolaris maydis (Helminthosporium maydis)",
                "Penyebaran spora melalui angin (airborne disease)",
                "Faktor lingkungan: kelembaban tinggi dan suhu 25-30°C"
            ],
            "gejala": [
                "Muncul bercak bulat memanjang berwarna hijau kekuningan hingga cokelat kemerahan",
                "Bercak dapat menyatu membentuk area kerusakan lebih besar",
                "Pada serangan berat, daun mengering dan mati"
            ],
            "pencegahan": [
                "Gunakan benih varietas unggul tahan penyakit",
                "Lakukan rotasi tanaman dengan non-inang (bukan jagung)",
                "Sanitasi lahan dari sisa tanaman terinfeksi"
            ],
            "penanganan": [
                "Aplikasi fungisida (mankozeb/klorotalonil)",
                "Pemantauan rutin untuk deteksi dini",
                "Pengaturan jarak tanam optimal (45-60 cm)"
            ],
           "referensi": [
                {
                    "text": "Pusat Penelitian Tanaman Pangan",
                    "url": "https://drive.google.com/file/d/1gPDJGDdQ65FMpkCBS9XejZenN8p7zk2v/view?usp=sharing"
                },
                {
                    "text": "Kunan Meneliti", 
                    "url": "https://drive.google.com/file/d/1GxwzvwifOtaoXeZoY2BxC2TB5x2RoqFh/view?usp=sharing"
                }
            ]
        },
        "Karat Daun": {
            "nama_latin": "Puccinia sorghi / Puccinia polysora",
            "penyebab": [
                "Disebabkan oleh cendawan Puccinia sorghi atau Puccinia polysora",
                "Spora menyebar melalui angin",
                "Perkembangan cepat di lingkungan lembab"
            ],
            "gejala": [
                "Bercak kecil berwarna cokelat, kuning, oranye, atau merah (uredinia)",
                "Bercak dikelilingi halo kuning",
                "Daun mengering saat serangan berat"
            ],
            "pencegahan": [
                "Pilih varietas tahan karat daun",
                "Hindari penanaman terlalu rapat",
                "Hindari pemupukan nitrogen berlebihan"
            ],
            "penanganan": [
                "Fungisida berbahan aktif triazol/strobilurin",
                "Cabut dan musnahkan tanaman terinfeksi parah",
                "Penyemprotan preventif musim hujan"
            ],
            "referensi": "Jurnal Penyakit Tanaman 2023"
        },
        "Hawar Daun": {
            "nama_latin": "Exserohilum turcicum",
            "penyebab": [
                "Disebabkan oleh cendawan Exserohilum turcicum",
                "Spora menyebar via angin/air hujan",
                "Optimal di kelembaban >80% & suhu 20-30°C"
            ],
            "gejala": [
                "Bercak memanjang hijau kekuningan sejajar tulang daun",
                "Bercak membesar jadi cokelat keabu-abuan dengan bagian tengah kering",
                "Serbuk hitam (konidia) di kelembaban tinggi"
            ],
            "pencegahan": [
                "Tanam varietas tahan (Kalingga, Arjuna, Pioneer 2)",
                "Rotasi tanaman dengan non-inang",
                "Tanam di awal/akhir musim kemarau"
            ],
            "penanganan": [
                "Fungisida mancozeb/klorotalonil/triazol",
                "Agens hayati (Pseudomonas/Trichoderma)",
                "Pemangkasan daun terinfeksi berat"
            ],
            "referensi": "Jurnal Penyakit Tanaman Jagung 2022"
        },
        "Daun Sehat": {
            "penyebab": [
                "Tidak ada infeksi patogen",
                "Kondisi lingkungan optimal",
                "Perawatan tanaman yang baik"
            ],
            "gejala": [
                "Warna hijau cerah merata",
                "Tekstur daun kuat tidak layu",
                "Bebas bintik/perubahan warna"
            ],
            "pencegahan": [
                "Pemupukan berimbang",
                "Pengairan cukup",
                "Pemantauan rutin"
            ],
            "penanganan": [
                "Pertahankan kondisi optimal",
                "Lakukan pencegahan penyakit",
                "Tidak perlu perlakuan khusus"
            ],
            "referensi": "Buku Perawatan Tanaman oleh Dr. Citra"
        }
    }
    info = disease_info.get(predicted_label, {})

    return render_template(
        'result.html',
        label=predicted_label,
        image_path=file_path,
        preprocessed_image=preprocessed_path,
        hsv_image=hsv_path,
        grayscale_image=grayscale_path,
        probabilities=class_probabilities,
        info=info,
        is_tebu=(predicted_label == "Tebu")  # 🔹 tambahan alert tebu
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
