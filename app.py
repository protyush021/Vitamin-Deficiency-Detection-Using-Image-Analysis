import os
import base64
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import dlib
import cv2
from datetime import datetime
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Custom-CNN model
main_model = tf.keras.models.load_model('saved_model/body_parts_model')
class_labels = ['Eye', 'Face', 'Skin', 'Nails', 'Tongue', 'Lips']

# VGG-16 model
models = {
    "Eye": tf.keras.models.load_model('Body Parts Saved Model/BodyParts_VGG16_saved_model_Eye'),
    "Face": tf.keras.models.load_model('Body Parts Saved Model/BodyParts_VGG16_saved_model_Face'),
    "Lips": tf.keras.models.load_model('Body Parts Saved Model/BodyParts_VGG16_saved_model_Lips'),
    "Nails": tf.keras.models.load_model('Body Parts Saved Model/BodyParts_VGG16_saved_model_Nails'),
    "Tongue": tf.keras.models.load_model('Body Parts Saved Model/BodyParts_VGG16_saved_model_Tongue')
}

class_labels_map = {
    "Eye": ['Bulging Eyes', 'Cataracts', 'Crossed Eyes', 'Glaucoma', 'Healthy Eye', 'Uveitis'],
    "Face": ['Acne or Breakouts', 'Clear Face', 'Dry or Flaky Skin', 'Hyperpigmentation or Dark Patches', 'Pale Skin'],
    "Lips": ['Angular cheilitis', 'Cracked', 'Normal'],
    "Nails": ["beau's line", 'black line', 'clubbing', "muehrck-e's lines", 'onycholysis', "terry's nail", 'white spot'],
    "Tongue": ['Magenta Coloured Tongue', 'Not Visible / Normal', 'Swollen Red', 'White Tongue']
}

medical_references = {
    "Eye": {
        "Bulging Eyes": "May indicate hyperthyroidism (Source: Mayo Clinic).",
        "Cataracts": "Linked to Vitamin A deficiency (Source: NIH).",
        "Crossed Eyes": "Not directly linked to vitamin deficiency.",
        "Glaucoma": "Not directly linked to vitamin deficiency.",
        "Healthy Eye": "No deficiency indicated.",
        "Uveitis": "May be linked to Vitamin D deficiency (Source: PubMed)."
    },
    "Face": {
        "Acne or Breakouts": "May indicate hormonal imbalance or poor diet (Source: Cleveland Clinic).",
        "Clear Face": "No deficiency indicated.",
        "Dry or Flaky Skin": "May indicate dehydration or Vitamin A deficiency (Source: NIH).",
        "Hyperpigmentation or Dark Patches": "May indicate sun exposure or hormonal changes (Source: Mayo Clinic).",
        "Pale Skin": "May indicate iron or Vitamin B12 deficiency (Source: Mayo Clinic)."
    },
    "Lips": {
        "Angular cheilitis": "Linked to Vitamin B2 or B12 deficiency (Source: NIH).",
        "Cracked": "May indicate dehydration or Vitamin B deficiency (Source: Cleveland Clinic).",
        "Normal": "No deficiency indicated."
    },
    "Nails": {
        "beau's line": "May indicate zinc deficiency (Source: NIH).",
        "black line": "Not directly linked to vitamin deficiency.",
        "clubbing": "Not directly linked to vitamin deficiency.",
        "muehrck-e's lines": "May indicate protein deficiency (Source: PubMed).",
        "onycholysis": "May indicate iron deficiency (Source: NIH).",
        "terry's nail": "May indicate liver issues or Vitamin B12 deficiency (Source: Mayo Clinic).",
        "white spot": "May indicate zinc deficiency (Source: Cleveland Clinic)."
    },
    "Tongue": {
        "Magenta Coloured Tongue": "Linked to Vitamin B12 deficiency (Source: NIH).",
        "Not Visible / Normal": "No deficiency indicated.",
        "Swollen Red": "May indicate Vitamin B3 deficiency (Source: PubMed).",
        "White Tongue": "May indicate Vitamin B12 or iron deficiency (Source: Cleveland Clinic)."
    }
}

# New dictionary for condition causes/characteristics
condition_causes = {
    "Eye": {
        "Bulging Eyes": ["Eyes protruding outward", "Possible thyroid-related swelling"],
        "Cataracts": ["Cloudy or opaque lens", "Blurred vision"],
        "Crossed Eyes": ["Misaligned eyes", "One or both eyes turn inward or outward"],
        "Glaucoma": ["Increased eye pressure", "Possible optic nerve damage"],
        "Healthy Eye": ["No redness", "No bulging", "Clear vision"],
        "Uveitis": ["Redness around iris", "Eye inflammation"]
    },
    "Face": {
        "Acne or Breakouts": ["Presence of pimples or cysts", "Inflammation and redness"],
        "Clear Face": ["Even skin tone", "No blemishes or acne"],
        "Dry or Flaky Skin": ["Rough texture", "Visible flakes or dryness"],
        "Hyperpigmentation or Dark Patches": ["Darkened areas on skin", "Uneven skin tone"],
        "Pale Skin": ["Lack of color", "Possible fatigue appearance"]
    },
    "Lips": {
        "Angular cheilitis": ["Cracks at corners of mouth", "Redness or inflammation"],
        "Cracked": ["Dry, split lip surface", "Rough texture"],
        "Normal": ["Smooth lip surface", "No cracks or redness"]
    },
    "Nails": {
        "beau's line": ["Horizontal ridges or grooves", "Nail growth interruption"],
        "black line": ["Dark vertical streaks", "Possible pigmentation issue"],
        "clubbing": ["Enlarged fingertips", "Curved nail shape"],
        "muehrck-e's lines": ["White horizontal bands", "Paired lines across nail"],
        "onycholysis": ["Nail lifting from bed", "Separation at tip"],
        "terry's nail": ["White nail bed", "Dark band at tip"],
        "white spot": ["Small white patches", "Irregular spots on nail"]
    },
    "Tongue": {
        "Magenta Coloured Tongue": ["Bright purple or magenta hue", "Smooth surface"],
        "Not Visible / Normal": ["Pink color", "No swelling", "Even texture"],
        "Swollen Red": ["Enlarged tongue", "Red and glossy appearance"],
        "White Tongue": ["White coating", "Possible patchy texture"]
    }
}

vitamin_data = pd.read_csv('vitamin_deficiency_combinations.csv')
vitamin_data.fillna("", inplace=True)  

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
CROPPED_FOLDER = 'static/cropped'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_vitamin_deficiency(detected_conditions):
    if "Hand" in detected_conditions:
        return (
            "No major Deficiencies found.",
            "Maintain a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats. "
            "Stay hydrated, exercise regularly, and ensure adequate sunlight exposure to prevent deficiencies and maintain good health."
        )

    match = vitamin_data.copy()
    
    print("\nDetected Conditions:")
    for part in ["Eye", "Face", "Lips", "Nails", "Tongue"]:
        condition = detected_conditions.get(part, "")
        print(f"{part} = '{condition}'")  
        match = match[match[part] == condition]

    print("\nFiltered DataFrame after applying conditions:")
    print(match)

    if match.empty:
        return "No major Deficiencies found.", (
            "Maintain a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats. "
            "Stay hydrated, exercise regularly, and ensure adequate sunlight exposure to prevent deficiencies and maintain good health."
        )

    return match.iloc[0]['Vitamin Deficiency'], match.iloc[0]['Recommendation']

def preprocess_image(image):
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def save_image_from_base64(base64_string):
    image_data = base64.b64decode(base64_string.split(',')[1])
    filename = f"captured_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    return filename, file_path

def crop_and_save(image, part_name, points):
    x_coords, y_coords = zip(*points)
    x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, image.width)
    y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, image.height)

    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    filename = f"{part_name.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    path = os.path.join(CROPPED_FOLDER, filename)
    cropped_image.save(path)
    return filename

def detect_body_parts(image, file_path):
    detected_parts, image_with_landmarks, cropped_images = detect_face_eyes_lips_tongue(image)

    if not detected_parts:
        img_array = preprocess_image(image)
        predictions = main_model.predict(img_array)[0]
        detected_parts = [class_labels[i] for i, pred in enumerate(predictions) if pred > 0.5]

    if not all(part in detected_parts for part in ['Face', 'Eye', 'Lips', 'Tongue']):
        for part in detected_parts:
            cropped_images[part] = file_path  

    return detected_parts, cropped_images

def detect_face_eyes_lips_tongue(image):
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    detected_parts = []
    cropped_images = {}
    draw = ImageDraw.Draw(image)

    for face in faces:
        if 'Face' not in detected_parts:
            detected_parts.append('Face')
        face_coords = [(face.left(), face.top()), (face.right(), face.bottom())]
        cropped_images['Face'] = crop_and_save(image, 'Face', face_coords)
        draw.rectangle(face_coords, outline='red', width=2)

        landmarks = predictor(gray, face)

        parts_map = {
            'Lips': [(48, 60)],
            'Tongue': [(60, 68)]
        }
        eye_crops = []  # Store multiple eye images
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        if left_eye_points:
            eye_crops.append(crop_and_save(image, 'Eye_Left', left_eye_points))
            draw.polygon(left_eye_points, outline='blue')

        if right_eye_points:
            eye_crops.append(crop_and_save(image, 'Eye_Right', right_eye_points))
            draw.polygon(right_eye_points, outline='blue')

        if eye_crops:  
            detected_parts.append('Eye')
            cropped_images['Eye'] = eye_crops  # Store both left and right eye images

        # Process Lips & Tongue
        for part_name in ['Lips', 'Tongue']:
            part_crops = []
            for start, end in parts_map[part_name]:
                part_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]
                if part_points:
                    part_crops.append(crop_and_save(image, part_name, part_points))
                    draw.polygon(part_points, outline='blue')

            if part_crops:
                detected_parts.append(part_name)
                cropped_images[part_name] = part_crops[0]

    return detected_parts, image, cropped_images

def compute_mean_rgb(image_path):
    """Compute the mean RGB values of an image."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    mean_rgb = np.mean(img_array, axis=(0, 1))  # Compute mean across height and width
    return mean_rgb

def predict_body_part(image_filename, part_name, uploaded_file_path):
    if part_name == "Skin":
        return "Nothing significant found", 0.0, [], (0, 0, 0)
    
    image_path = os.path.join(CROPPED_FOLDER, image_filename) if os.path.exists(os.path.join(CROPPED_FOLDER, image_filename)) else uploaded_file_path
    
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return "File Not Found", 0.0, [], (0, 0, 0)

    # Compute mean RGB for color analysis
    mean_rgb = compute_mean_rgb(image_path)

    img = Image.open(image_path).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    model = models.get(part_name)
    
    if model:
        predictions = model.predict(img_array)[0]
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        confidence_scores = [(label, float(pred)) for label, pred in zip(class_labels_map[part_name], predictions)]
        return class_labels_map[part_name][class_idx], float(confidence), confidence_scores, mean_rgb
    return "Unknown", 0.0, [], (0, 0, 0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_path = None
        filename = None
        
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = file.filename
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
        elif 'captured_image' in request.form:
            filename, file_path = save_image_from_base64(request.form['captured_image'])

        if file_path:
            image = Image.open(file_path).convert('RGB')
            detected_parts, cropped_images = detect_body_parts(image, file_path)
            results = {part: predict_body_part(cropped_images[part] if isinstance(cropped_images[part], str) else cropped_images[part][0], part, file_path) for part in detected_parts}
            
            detected_conditions = {part: result[0] for part, result in results.items()}
            vitamin_deficiency, recommendations = get_vitamin_deficiency(detected_conditions)

            return render_template(
                'result.html',
                uploaded_image=filename,
                parts=detected_parts,
                cropped_images=cropped_images,
                results=results,
                vitamin_deficiency=vitamin_deficiency,
                recommendations=recommendations,
                medical_references=medical_references,
                condition_causes=condition_causes  # Pass the new dictionary
            )
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/cropped/<filename>')
def cropped_file(filename):
    return send_from_directory(CROPPED_FOLDER, filename)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

if __name__ == '__main__':
    app.run(debug=True)