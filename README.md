# Pokémon Image Classifier

A deep-learning based image classification application that identifies Pokémon from uploaded images using **transfer learning with MobileNetV2**. The project includes a Streamlit interface for interactive image classification and displays the predicted Pokémon along with its confidence score.

## Features

* 🧠 Pokémon image classification using **MobileNetV2**
* 🔄 Transfer learning with a pretrained CNN architecture
* 🖼️ Image upload and preprocessing for model inference
* 📊 Prediction with confidence score
* 🌐 Interactive **Streamlit** web interface
* 📚 Pokémon information displayed alongside predictions

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **Pillow**
* **Streamlit**

## Project Structure

```text
Pokemon-Image-Classifier/
│
├── app_poke.py                  # Streamlit application
├── model.py                     # Model architecture and training
├── Work.py                      # Supporting project code
├── myMLmodel_mobilenet.keras    # Trained MobileNetV2 model
├── pokemon-party.jpg            # Sample image
├── requirements.txt             # Python dependencies
└── README.md
```

## Model

The project uses **MobileNetV2** with transfer learning for image classification.

The pretrained convolutional layers are used as a feature extractor, followed by custom classification layers for Pokémon recognition. Image augmentation techniques such as rotation, zoom, and horizontal flipping are applied during training to improve model generalization.

The classifier predicts across **150 Pokémon categories**.

## Installation

Clone the repository:

```bash
git clone https://github.com/mishkat-fatma/Pokemon-Image-Classifier.git
cd Pokemon-Image-Classifier
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Launch the Streamlit application:

```bash
streamlit run app_poke.py
```

The application will open in your browser. Upload a Pokémon image to receive the model's predicted class and confidence score.

## Workflow

```text
Input Image
     ↓
Image Preprocessing
     ↓
MobileNetV2 Feature Extraction
     ↓
Classification Layer
     ↓
Predicted Pokémon + Confidence
```

## Future Improvements

* Improve classification accuracy with a larger and more diverse dataset
* Add model evaluation metrics such as precision, recall, and F1-score
* Expand the classifier to additional Pokémon categories
* Improve the Streamlit interface with richer Pokémon information
* Optimize inference performance for deployment

## Disclaimer

This project was developed for educational and experimental purposes to demonstrate **deep learning, transfer learning, computer vision, and interactive ML application development**.
