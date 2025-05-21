# Python ASL Translator
A real-time American Sign Language (ASL) translator leveraging Convolutional Neural Networks (CNNs) to interpret hand gestures captured via webcam and convert them into corresponding English text.​

This project aims to bridge communication gaps for the deaf and hard-of-hearing community by translating ASL gestures into readable text. Utilizing computer vision and deep learning techniques, the system processes live video input to recognize ASL alphabets.​

## 📂 Project Structure
- `asl_dataset/`: Directory containing ASL images used for training and validation.
- `asl_cnn_model.keras`: Pre-trained Keras model for ASL alphabet recognition.
- `cam.py`: Script to capture real-time video input from the webcam.
- `main.py`: Main application script that integrates video capture and model prediction.
- `requirements.txt`: List of Python dependencies required to run the project

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Webcam for capturing hand gestures​

Clone the Repository:
```bash
git clone https://github.com/NamanChandok/asl-translator.git && cd asl-translator
```

Install Dependencies:
```bash
pip install -r requirements.txt
```

Run the Python File
```bash
python3 Cam.py
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to the project, please fork the repository and submit a pull request.
