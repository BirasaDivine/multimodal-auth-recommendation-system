"""
============================================================
  User Identity & Product Recommendation System
  Command-Line Application
============================================================

  FLOW:
    START
      │
      ▼
    [STEP 1] Face Recognition
      │ FAIL → Access Denied (exit)
      │ PASS
      ▼
    [STEP 2] Product Recommendation Model  ← computes but does NOT display yet
      │
      ▼
    [STEP 3] Voice Verification
      │ FAIL → Access Denied, prediction discarded (exit)
      │ PASS
      ▼
    Display Predicted Product

  HOW THE VOICE MODEL WORKS:
    The voiceprint model is a SPEAKER IDENTIFICATION model.
    It was trained to recognise 4 speakers by name:
      Divine, Emma, Florence, Yinka
    "Authorized" means the predicted speaker matches the
    expected speaker for the given face (e.g., face=Florence
    → voice must also be identified as Florence).
    If the predicted speaker does NOT match → Access Denied.

  Usage:
    python app.py --face "images/florence/smile.jpg" \\
                  --voice "Florence 2.wav" \\
                  --customer_id 150 \\
                  --speaker Florence

    python app.py --simulate_unauthorized --type face
    python app.py --simulate_unauthorized --type voice --speaker Florence

  Requirements:
    pip install numpy pandas scikit-learn xgboost librosa
                opencv-python joblib scikit-image
============================================================
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Optional imports ─────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python not installed.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARNING] librosa not installed.")

try:
    from skimage.feature import hog, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image not installed.")


# ============================================================
#  SECTION 1: MODEL & DATA PATHS
#  Make sure to update file paths to match your file location
#  if repo is cloned
# ============================================================

FACE_MODEL_PATH    = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\models\face_recognition_model.pkl"

# Voiceprint model — saved by Voiceprint_Verification.ipynb
VOICE_MODEL_PATH         = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\models\best_voice_model.pkl"
VOICE_SCALER_PATH        = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\Voiceprint_verification\voice_scaler.pkl"
VOICE_LABEL_ENCODER_PATH = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\Voiceprint_verification\voice_label_encoder.pkl"

# Product model — saved by Prediction_model2.ipynb
PRODUCT_MODEL_PATH = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\models\product_recommendation_model.pkl"
LABEL_ENCODER_PATH = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\models\label_encoder.pkl"
MERGED_DATA_PATH   = r"C:\Users\Lenovo\OneDrive\Documents\ML_pipeline_assignments\multimodal-auth-recommendation-system\image_verification\data\merged_dataset.csv"


IMG_SIZE = (128, 128)


SR     = 22050   
N_MFCC = 13      

# The 4 authorized speakers (must match label_encoder classes in the notebook)
AUTHORIZED_SPEAKERS = {'Divine', 'Emma', 'Florence', 'Yinka'}

# Maps image folder names → speaker names known by the voice model

FOLDER_TO_SPEAKER = {
    'birassa':    'Divine',
    'florence':   'Florence',
    'yinka':      'Yinka',
    'emmanuella': 'Emma',
}

# Product model platform columns 
PLATFORM_COLUMNS = [
    'social_media_platform_Facebook',
    'social_media_platform_Instagram',
    'social_media_platform_LinkedIn',
    'social_media_platform_TikTok',
    'social_media_platform_Twitter',
]


# ============================================================
#  SECTION 2: STUB MODEL
#  Runs automatically if no file path is found or incorrect
# ============================================================

class StubModel:
    """Placeholder used when a real .pkl is not found."""
    def __init__(self, name):
        self.name = name

    def predict(self, X):
        print(f"  [STUB] {self.name} — placeholder active (.pkl not found)")
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.15, 0.85]])


def load_model(path, name):
    if os.path.exists(path):
        print(f"  [OK] Loaded: {name}")
        return joblib.load(path)
    print(f"  [STUB] '{path}' not found — using placeholder for: {name}")
    return StubModel(name)


def load_artifact(path, name):
    """Load a non-model artifact (scaler, label encoder, etc.)"""
    if os.path.exists(path):
        print(f"  [OK] Loaded: {name}")
        return joblib.load(path)
    print(f"  [WARN] '{path}' not found — {name} unavailable.")
    return None


# ============================================================
#  SECTION 3: IMAGE FEATURE EXTRACTION
#  Matches face_recognition_improved.py exactly (HOG + LBP).
# ============================================================

def extract_hog_features(img_rgb):
    img_gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    features, _ = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    return features


def extract_lbp_features(img_rgb, num_points=24, radius=3):
    img_gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    lbp    = local_binary_pattern(img_resized, num_points, radius, method='uniform')
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_image_features(image_path):
    """Returns HOG + LBP feature vector for a face image."""
    if CV2_AVAILABLE and SKIMAGE_AVAILABLE and os.path.exists(image_path):
        img     = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = np.concatenate([
            extract_hog_features(img_rgb),
            extract_lbp_features(img_rgb)
        ])
        return features.reshape(1, -1)
    else:
        if not os.path.exists(image_path):
            print(f"  [ERROR] Image file not found: {image_path}")
        return np.random.rand(1, 8126)


# ============================================================
#  SECTION 4: AUDIO FEATURE EXTRACTION
#
#  Matches Voiceprint_Verification.ipynb extract_features() exactly:
#    - Load audio at SR=22050
#    - 13 MFCC means
#    - 1  spectral roll-off mean
#    - 1  RMS energy mean
#    = 15 features total
#  Then scale with voice_scaler before predicting.
# ============================================================

def extract_audio_features(audio_path):
    """
    Extracts 15 features from an audio file.
    Matches the extract_features() function in Voiceprint_Verification.ipynb.

    NOTE: The returned array is RAW (unscaled).
    The scaler must be applied separately before calling model.predict().
    This mirrors how the notebook works:
      scaler.fit_transform(X_train) then scaler.transform(X_test)
    """
    if not LIBROSA_AVAILABLE:
        print("  [ERROR] librosa is not installed.")
        return None

    if not os.path.exists(audio_path):
        print(f"  [ERROR] Audio file not found: {audio_path}")
        print(f"  Make sure the path is correct and the file exists.")
        return None

    try:
       
        y, sr = librosa.load(audio_path, sr=SR)

        
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=1)

        
        rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        
        rms_mean = np.mean(librosa.feature.rms(y=y))

        
        features = {f'mfcc_{i+1}': mfccs_mean[i] for i in range(13)}
        features['spectral_rolloff'] = rolloff_mean
        features['rms_energy']       = rms_mean

        # Return as 2D array in the correct column order
        feature_array = np.array(list(features.values())).reshape(1, -1)
        return feature_array

    except Exception as e:
        print(f"  [ERROR] Could not process audio file: {e}")
        return None


# ============================================================
#  SECTION 5: PRODUCT RECOMMENDATION FEATURE BUILDER
#
#  Matches Prediction_model2.ipynb exactly:
#  - Reads from merged_dataset.csv (already encoded + scaled)
#  - Feature columns: customer_id_new, engagement_score,
#    purchase_interest_score, review_sentiment, purchase_amount,
#    month, year, dayofweek, + 5 platform dummies
# ============================================================

def get_customer_features(customer_id, data_path):
    """
    Looks up the customer in merged_dataset.csv.
    The CSV was saved after all encoding and scaling was applied,
    so values are already in the correct form for model.predict().
    """
    if not os.path.exists(data_path):
        print(f"  [ERROR] '{data_path}' not found.")
        return None

    df = pd.read_csv(data_path)

    match = df[df['customer_id_new'] == customer_id]
    if match.empty:
        print(f"  [WARNING] Customer ID {customer_id} not found in dataset.")
        return None

    # Use the most recent transaction row
    row = match.iloc[-1]

    feature_cols = [
        'customer_id_new',
        'engagement_score',
        'purchase_interest_score',
        'review_sentiment',
        'purchase_amount',
        'month',
        'year',
        'dayofweek',
    ] + PLATFORM_COLUMNS

    feature_dict = {}
    for col in feature_cols:
        feature_dict[col] = row[col] if col in row.index else 0

    return pd.DataFrame([feature_dict])[feature_cols]


def decode_product(raw_pred, label_encoder):
    """Converts numeric prediction back to product category name."""
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([int(raw_pred)])[0]
        except Exception:
            pass
    fallback = {0: 'Books', 1: 'Clothing', 2: 'Electronics',
                3: 'Groceries', 4: 'Sports'}
    return fallback.get(int(raw_pred), str(raw_pred))


# ============================================================
#  SECTION 6: THE THREE PIPELINE STEPS
# ============================================================

def step_face_recognition(image_path, model):
    """
    STEP 1 — Face Recognition.
    Model predicts 1 (recognized) or 0 (not recognized).
    FAIL → Access Denied.
    """
    print("\n─────────────────────────────────────────────")
    print("  STEP 1: Facial Recognition")
    print("─────────────────────────────────────────────")
    print(f"  Input       : {image_path}")

    features   = extract_image_features(image_path)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1] * 100

    if prediction == 1:
        print(f"  Result      : RECOGNIZED ")
        print(f"  Confidence  : {confidence:.1f}%")
        print("  → Face authentication PASSED. Running product model...")
        return True
    else:
        print(f"  Result      : NOT RECOGNIZED ")
        print(f"  Confidence  : {confidence:.1f}%")
        _print_denied("Face not recognized.")
        return False


def step_product_recommendation(customer_id, model, data_path, label_encoder):
    """
    STEP 2 — Product Recommendation.
    Runs after face passes. Result held until voice passes.
    """
    print("\n─────────────────────────────────────────────")
    print("  STEP 2: Product Recommendation (computing...)")
    print("─────────────────────────────────────────────")
    print(f"  Customer ID : {customer_id}")

    features = get_customer_features(customer_id, data_path)
    if features is None:
        raw_pred = model.predict(np.random.rand(1, 13))[0]
    else:
        raw_pred = model.predict(features)[0]

    product = decode_product(raw_pred, label_encoder)
    print(f"  Status      : Prediction computed. Awaiting voice verification...")
    print(f"  (Result held — will display after voice check passes)")
    return product


def step_voice_verification(audio_path, model, scaler, voice_label_encoder,
                             expected_speaker):
    """
    STEP 3 - Voice Verification (strict face-voice match).

    The voiceprint model predicts WHO is speaking.
    VERIFIED only if predicted speaker == expected speaker from face.
    This prevents using your own voice with someone else's face.
    """
    print("\n─────────────────────────────────────────────")
    print("  STEP 3: Voiceprint Verification")
    print("─────────────────────────────────────────────")
    print(f"  Input            : {audio_path}")
    print(f"  Expected speaker : {expected_speaker}")

    features = extract_audio_features(audio_path)
    if features is None:
        print("  Result      : FAILED (could not load audio) ")
        _print_denied("Audio file could not be processed.")
        return False

    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        print("  [WARN] No scaler loaded — using unscaled features")
        features_scaled = features

    raw_pred = model.predict(features_scaled)[0]

    if voice_label_encoder is not None:
        try:
            predicted_speaker = voice_label_encoder.inverse_transform([int(raw_pred)])[0]
        except Exception:
            predicted_speaker = str(raw_pred)
    else:
        predicted_speaker = str(raw_pred)

    try:
        proba      = model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
    except Exception:
        confidence = 0.0

    print(f"  Predicted speaker: {predicted_speaker}")
    print(f"  Confidence       : {confidence:.1f}%")

    # Strict match: voice must belong to the same person as the face
    if predicted_speaker.strip().lower() == expected_speaker.strip().lower():
        print(f"  Result      : VERIFIED   ({predicted_speaker} matches face identity)")
        print("  → Voice authentication PASSED. Displaying prediction...")
        return True
    else:
        print(f"  Result      : NOT VERIFIED ")
        print(f"  Voice identified as '{predicted_speaker}' but face belongs to '{expected_speaker}'")
        _print_denied("Voice/face mismatch detected.")
        return False


def display_product(product_label):
    """FINAL — Display recommendation after both checks pass."""
    print("\n╔════════════════════════════════════════════╗")
    print("    PREDICTION APPROVED")
    print(f"  ★   Recommended product: {product_label}")
    print("╚════════════════════════════════════════════╝")


def _print_denied(reason):
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║         ACCESS DENIED                ║")
    print(f"  ║  {reason:<36}║")
    print(f"  ╚══════════════════════════════════════╝")


# ============================================================
#  SECTION 7: UNAUTHORIZED SIMULATION
# ============================================================

def simulate_unauthorized(sim_type, face_model, voice_model,
                           voice_scaler, voice_label_encoder,
                           expected_speaker="Florence"):
    """
    sim_type = 'face'  → unknown face rejected at Step 1
    sim_type = 'voice' → voice/face mismatch rejected at Step 3
                         (simulates someone using the wrong person's voice)
    """
    print("\n╔═══════════════════════════════════════════════╗")
    print("║     SIMULATING UNAUTHORIZED ATTEMPT           ║")
    print("╚═══════════════════════════════════════════════╝")

    if sim_type == "face":
        print("\n  Presenting an UNKNOWN face (blank/zero image)...")
        fake_features = np.zeros((1, 8126))
        prediction    = face_model.predict(fake_features)[0]

        print("\n  STEP 1: Facial Recognition")
        if prediction == 0:
            print("  Result : NOT RECOGNIZED ")
            _print_denied("Unknown face rejected.")
            print("\n  ✓ System correctly blocked unauthorized face.")
        else:
            print("  [NOTE] Stub/untrained model approved this.")
            print("         Use your real face_recognition_model.pkl.")

    elif sim_type == "voice":
        print("\n  Face check PASSED (simulated).")
        print("  Product model computed (simulated).")
        print("\n  Presenting UNKNOWN voice (silence/zero audio)...")

        print("\n  STEP 3: Voiceprint Verification")
        fake_features = np.zeros((1, 15))

        if voice_scaler is not None:
            fake_scaled = voice_scaler.transform(fake_features)
        else:
            fake_scaled = fake_features

        raw_pred = voice_model.predict(fake_scaled)[0]

        if voice_label_encoder is not None:
            try:
                predicted_speaker = voice_label_encoder.inverse_transform([int(raw_pred)])[0]
            except Exception:
                predicted_speaker = str(raw_pred)
        else:
            predicted_speaker = str(raw_pred)

        print(f"  Predicted speaker : {predicted_speaker}")
        print(f"  Expected speaker  : {expected_speaker}")

        if predicted_speaker.strip().lower() != expected_speaker.strip().lower():
            print(f"  Result : NOT VERIFIED ")
            _print_denied(f"Voice/face mismatch detected.")
            print("\n   System correctly blocked voice/face mismatch.")
        else:
            print("  [NOTE] Simulation matched — for a proper mismatch demo,")
            print("         use a recording from a different speaker.")

    print("\n  Simulation complete.")


# ============================================================
#  SECTION 8: MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="User Identity & Product Recommendation System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--face",        type=str,
                        help="Path to face image (.jpg / .png)")
    parser.add_argument("--voice",       type=str,
                        help="Path to voice sample (.wav)")
    parser.add_argument("--customer_id", type=int,
                        help="Numeric customer ID (e.g. 150)")
    parser.add_argument("--simulate_unauthorized", action="store_true",
                        help="Run an unauthorized access simulation")
    parser.add_argument("--type", type=str, choices=["face", "voice"],
                        help="Which step to simulate as unauthorized")
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║   User Identity & Product Recommendation System  ║")
    print("║   HOG+LBP Face  │  MFCC Voice  │  XGBoost+SMOTE  ║")
    print("╚═══════════════════════════════════════════════════╝")

    # ── Load models ───────────────────────────────────────────
    print("\n[LOADING MODELS]")
    face_model    = load_model(FACE_MODEL_PATH,    "Face Recognition (HOG+LBP)")
    voice_model   = load_model(VOICE_MODEL_PATH,   "Voiceprint Speaker ID")
    product_model = load_model(PRODUCT_MODEL_PATH, "Product Recommendation (XGBoost+SMOTE)")

    # ── Load supporting artifacts ─────────────────────────────
    voice_scaler          = load_artifact(VOICE_SCALER_PATH,        "Voice scaler")
    voice_label_encoder   = load_artifact(VOICE_LABEL_ENCODER_PATH, "Voice label encoder")
    product_label_encoder = load_artifact(LABEL_ENCODER_PATH,       "Product label encoder")

    if voice_label_encoder is not None:
        print(f"         Voice classes: {list(voice_label_encoder.classes_)}")
    if product_label_encoder is not None:
        print(f"         Product classes: {list(product_label_encoder.classes_)}")

    # ── Route: simulation ────────────────────────────────────
    if args.simulate_unauthorized:
        if not args.type:
            print("\n[ERROR] Add --type face  or  --type voice")
            sys.exit(1)
        simulate_unauthorized(args.type, face_model, voice_model,
                              voice_scaler, voice_label_encoder,
                              expected_speaker="Florence")
        sys.exit(0)

    # ── Route: full run ───────────────────────────────────────
    if not args.face or not args.voice or args.customer_id is None:
        print("\n[ERROR] Provide --face, --voice, and --customer_id for a full run.")
        print("\n  Examples:")
        print('    python app.py --face "images/florence/smile.jpg" ^')
        print('                  --voice "Divine 1.wav" ^')
        print('                  --customer_id 150')
        print()
        print('    python app.py --simulate_unauthorized --type face')
        print('    python app.py --simulate_unauthorized --type voice')
        sys.exit(1)

    # ── Infer expected speaker from face image folder ────────
    # e.g.  .../raw-images/birassa/neutral.jpg  →  Divine
    def infer_speaker_from_path(face_path):
        parts = face_path.replace("\\", "/").split("/")
        for part in parts:
            if part.lower() in FOLDER_TO_SPEAKER:
                return FOLDER_TO_SPEAKER[part.lower()]
        return None

    expected_speaker = infer_speaker_from_path(args.face)
    if expected_speaker is None:
        print("\n[ERROR] Could not determine which speaker this face belongs to.")
        print("  Make sure the face image is inside a folder named after the person.")
        print("  Known folder names:")
        for folder, speaker in FOLDER_TO_SPEAKER.items():
            print(f"    {folder}  →  {speaker}")
        sys.exit(1)

    print(f"\n  Identity inferred from face path: {expected_speaker}")

    # ── STEP 1: Face Recognition ─────────────────────────────
    face_passed = step_face_recognition(args.face, face_model)
    if not face_passed:
        print("\n   Session ended: access denied at face recognition.")
        sys.exit(0)

    # ── STEP 2: Product Recommendation (compute, hold) ───────
    predicted_product = step_product_recommendation(
        customer_id   = args.customer_id,
        model         = product_model,
        data_path     = MERGED_DATA_PATH,
        label_encoder = product_label_encoder
    )

    # ── STEP 3: Voice Verification (must match face identity) ─
    voice_passed = step_voice_verification(
        audio_path          = args.voice,
        model               = voice_model,
        scaler              = voice_scaler,
        voice_label_encoder = voice_label_encoder,
        expected_speaker    = expected_speaker
    )
    if not voice_passed:
        print("\n  Session ended: access denied at voice verification.")
        print("     Prediction has been discarded.")
        sys.exit(0)

    # ── FINAL: Display prediction ─────────────────────────────
    display_product(predicted_product)

    print("\n═══════════════════════════════════════════════════")
    print("  Session complete.")
    print("═══════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()