# Multimodal Authentication and Product Recommendation System

**Project:** Formative 2 - Multimodal Data Preprocessing Assignment  

## Project Overview
This project implements a secure, sequential multimodal authentication pipeline that validates a user's identity before providing personalized product recommendations. It simulates a high-security environment where sensitive data (transactions and recommendations) is protected by a two-factor biometric lock.

The system relies on three distinct machine learning models:
1. **Visual Checkpoint (Facial Recognition):** Authenticates the user's face.
2. **Vocal Checkpoint (Voiceprint Verification):** Authenticates the user's voice via a spoken passphrase.
3. **Execution (Product Recommendation):** Recommends products based on merged social and transactional data. This is only unlocked if both biometric checks pass and match the exact same identity.

---

##  Repository Structure

```text
├── Voiceprint_verification/
│   ├── Divine 1.ogg / Divine 1.wav (and other raw audio files)
│   ├── Voiceprint_Verification.ipynb      # Phase 3: Audio modeling
│   ├── audio_features.csv                 # Extracted audio features
│   ├── voice_label_encoder.pkl
│   └── voice_scaler.pkl
├── image_verification/
│   ├── data/                              # Raw facial images
│   ├── Multimodal_Data_Preprocessing_Assignment (1).ipynb # Phase 2: Image modeling
│   ├── image_features.csv                 # Extracted color histogram features
│   ├── image_label_encoder.pkl
│   └── image_recognition_model.pkl
├── models/
│   ├── best_voice_model.pkl               # Final saved voice classifier
│   ├── face_recognition_model.pkl         # Final saved face classifier
│   ├── label_encoder.pkl                  # Product recommendation encoder
│   ├── product_recommendation_model.pkl   # Final saved product classifier
│   └── scaler.pkl                         # Product recommendation scaler
├── Prediction_Model.ipynb                 # Phase 1: Tabular data exploration
├── Prediction_model2.ipynb                # Phase 1: Final tabular merge and modeling
├── app.py                                 # The command-line app running the system
└── README.md                              # Project documentation

## Technical Pipeline
**Phase 1:** Tabular Data (Product Recommendation)
Data Merging: Merged customer_social_profiles and customer_transactions using an inner join to preserve data integrity.

Exploratory Data Analysis (EDA): Handled null values, fixed data types, and generated visualizations (distribution plots, outliers, correlation heatmaps) to analyze purchasing behavior.

Modeling: Evaluated and trained classifiers (Random Forest, XGBoost) to predict the most likely product purchase based on user demographics and social behavior. Saved the optimal model as product_recommendation_model.pkl.

**Phase 2:** Image Processing (Facial Recognition)
- **Data Collection:** 3 base images per team member (Neutral, Smiling, Surprised).

- **Augmentation:** Applied horizontal flipping, 90-degree rotation, and grayscale conversion using cv2 to artificially expand the dataset and improve model robustness.

- **Feature Extraction:** Extracted a 512-dimensional feature vector per image using a 3D RGB Color Histogram (8x8x8 bins).

- **Modeling:** Evaluated Random Forest, Logistic Regression, and XGBoost based on Accuracy, F1-Score, and Log Loss. The optimal model was saved as image_recognition_model.pkl.

**Phase 3:** Audio Processing (Voiceprint Verification)
Data Collection: 2 audio samples per team member speaking specific passphrases ("Yes, approve" and "Confirm transaction").

- **Augmentation:** Applied a 2-semitone pitch shift and a 1.2x time stretch using librosa to simulate natural variations in vocal tone and speaking speed.

- **Feature Extraction:** Extracted 15 engineered features per clip (13 MFCCs, Spectral Roll-off, and RMS Energy).

- **Modeling:** Scaled features using StandardScaler and trained a Logistic Regression classifier, which achieved the lowest Log Loss due to its efficiency in finding linear boundaries in small tabular datasets. Saved as best_voice_model.pkl.

**Phase 4:** System Integration & Multimodal Logic
The final pipeline operates on strict sequential logic via app.py:

The user submits a test image. The Facial Recognition model predicts Face_ID. If unknown, the system outputs Access Denied.

If the face is recognized, the user submits a voice sample. The Voiceprint model predicts Voice_ID.

Cross-Check: The system compares Face_ID and Voice_ID. If they do not match perfectly, it flags a spoofing attempt and outputs Access Denied.

Only upon a verified multimodal match does the system query the tabular dataset and output the user's personalized product recommendation.

## How to Run the Simulation
1. Install Dependencies
Ensure you have Python installed, then install the required libraries:

'pip install numpy pandas scikit-learn xgboost opencv-python librosa joblib'
2. Run the Command-Line App
Execute the main simulation script from the root directory.

To test an Authorized flow (Matching Biometrics):

'python app.py --face "data/raw-images/yinka/neutral.jpeg" --voice "data/raw-audio/Yinka_1.ogg" --customer_id 150 --speaker yinka'

To test an Unauthorized flow (Spoofing/Mismatched Biometrics):

'python app.py --simulate_unauthorized --type voice --speaker florence# multimodal-auth-recommendation-system'
