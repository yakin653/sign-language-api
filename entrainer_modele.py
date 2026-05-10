"""
🇹🇳 ENTRAÎNEMENT DU MODÈLE - LANGUE DES SIGNES TUNISIENNE
Lance ce script UNE FOIS pour créer modele_tunisien.pkl
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # pip install tqdm  (optionnel mais pratique)

# ==========================================
# CONFIG - MODIFIE CE CHEMIN !
# ==========================================
DATASET_PATH = "C:/Users/yakin/First-ever-Tunisian-Sign-Language-Dataset/Data"

# ^ Mets le chemin exact de ton dataset ici

OUTPUT_MODEL  = "modele_tunisien.pkl"
OUTPUT_LABELS = "labels_tunisiens.json"

# ==========================================
# MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # True pour les images fixes !
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extraire_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        
        # Normalisation par rapport au poignet (point 0)
        base_x = lm.landmark[0].x
        base_y = lm.landmark[0].y
        base_z = lm.landmark[0].z
        
        features = []
        for point in lm.landmark:
            features.extend([
                point.x - base_x,
                point.y - base_y,
                point.z - base_z
            ])
        return np.array(features)
    
    return None

# ==========================================
# COLLECTER LES DONNÉES
# ==========================================
print("=" * 60)
print("🇹🇳 ENTRAÎNEMENT DU MODÈLE TUNISIEN")
print("=" * 60)
print(f"\n📁 Dataset: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERREUR: Le dossier n'existe pas: {DATASET_PATH}")
    print("   Modifie la variable DATASET_PATH dans ce script !")
    exit(1)

X = []  # features
y = []  # labels
labels_set = set()

# Parcourir toutes les catégories et mots
categories = [d for d in os.listdir(DATASET_PATH) 
              if os.path.isdir(os.path.join(DATASET_PATH, d))]

print(f"\n📂 Catégories trouvées: {categories}")
total_images = 0
images_ok = 0
images_ko = 0

for cat in sorted(categories):
    cat_path = os.path.join(DATASET_PATH, cat)
    mots = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
    
    for mot in sorted(mots):
        mot_path = os.path.join(cat_path, mot)
        label = f"{cat}_{mot}"
        labels_set.add(label)
        
        images = [f for f in os.listdir(mot_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"   📸 {label}: {len(images)} images", end="")
        
        mot_ok = 0
        for img_file in images:
            img_path = os.path.join(mot_path, img_file)
            features = extraire_landmarks(img_path)
            
            if features is not None:
                X.append(features)
                y.append(label)
                mot_ok += 1
                images_ok += 1
            else:
                images_ko += 1
            
            total_images += 1
        
        print(f" → {mot_ok} détectées ✅")

print(f"\n📊 RÉSUMÉ:")
print(f"   Total images  : {total_images}")
print(f"   Mains détectées: {images_ok} ({100*images_ok//max(total_images,1)}%)")
print(f"   Non détectées : {images_ko}")
print(f"   Classes       : {len(labels_set)}")

if images_ok < 10:
    print("\n❌ Pas assez d'images détectées. Vérifie le chemin du dataset.")
    exit(1)

# ==========================================
# ENTRAÎNER LE MODÈLE
# ==========================================
X = np.array(X)
y = np.array(y)
labels_list = sorted(list(labels_set))

print(f"\n🧠 Entraînement du modèle...")
print(f"   Shape X: {X.shape}")
print(f"   Nb classes: {len(labels_list)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# ÉVALUATION
# ==========================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ ACCURACY: {acc*100:.1f}%")
print("\n📋 Rapport détaillé:")
print(classification_report(y_test, y_pred))
# ==========================================
# SAUVEGARDER
# ==========================================
joblib.dump(model, OUTPUT_MODEL)
print(f"\n💾 Modèle sauvegardé: {OUTPUT_MODEL}")

with open(OUTPUT_LABELS, 'w', encoding='utf-8') as f:
    json.dump(labels_list, f, ensure_ascii=False, indent=2)
print(f"💾 Labels sauvegardés: {OUTPUT_LABELS}")

print("\n🎉 ENTRAÎNEMENT TERMINÉ !")
print(f"   Lance maintenant: python traducteur_complet.py")

hands.close()