document.querySelector('form').addEventListener('submit', function (e) {
    const fileInput = document.querySelector('input[type="file"]');
    if (!fileInput.files.length) {
        alert('Silakan pilih gambar terlebih dahulu!');
        e.preventDefault();
    }
});