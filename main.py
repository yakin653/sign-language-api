"""
🇹🇳 API FastAPI - Traducteur Langue des Signes Tunisien
Déploiement sur Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import json
import os

app = FastAPI(
    title="Traducteur LST API",
    description="API de traduction de la langue des signes tunisienne",
    version="1.0.0"
)

# CORS - autorise Flutter à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# ==========================================
MODEL_PATH  = os.getenv("MODEL_PATH",  "modele_tunisien.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "labels_tunisiens.json")

model  = None
labels = []

@app.on_event("startup")
async def load_model():
    global model, labels
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Modèle non trouvé: {e}")

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"✅ Labels chargés: {len(labels)} classes")
    except Exception as e:
        print(f"⚠️ Labels non trouvés: {e}")


# ==========================================
# DICTIONNAIRE ARABIZI → ARABE
# ==========================================
TRADUCTION_ARABE = {
    "3aslema": "عسلامة", "5adamet": "خدامت", "assam": "اسمع",
    "barnamjk": "برنامجك", "chabeb": "شباب", "cv": "سي في",
    "demande": "طلب", "enti": "انتي", "labes": "لاباس",
    "lyoum": "اليوم", "mar7ba": "مرحبا", "n3awnek": "نعاونك",
    "nekteblk": "نكتبلك", "non": "لا", "oui": "آه",
    "radio": "راديو", "se7a": "صحة", "siye7a": "سياحة",
    "t7eb": "تحب", "ta3lim": "تعليم", "ta3raf": "تعرف",
    "ta9ra": "تقرا", "telvza": "تلفزة", "tha9afa": "ثقافة",
    "baladya": "بلدية", "banka": "بنكة", "bousta": "بوسطة",
    "dar": "دار", "ma7kma": "محكمة", "mostawsaf": "مستوصف",
    "sbitar": "سبيطار", "wzara": "وزارة", "3ayla": "عيلة",
    "5al-3am": "خال عم", "5ou": "خو", "bent": "بنت",
    "bou": "بو", "eben": "ابن", "jad": "جد", "jadda": "جدة",
    "mar2a": "مرأة", "o5t": "أخت", "om": "أم", "tfol": "طفل",
    "5mis": "خميس", "a7ad": "احد", "erb3a": "اربعاء",
    "jom3a": "جمعة", "sebt": "سبت", "thleth": "ثلاثاء",
    "thnin": "اثنين", "car": "كار", "karhba": "كرهبة",
    "louage": "لواج", "métro": "مترو", "taxi": "تاكسي",
    "train": "تران",
}

LABEL_TO_ARABIZI = {
    "Demandes_3aslema": "3aslema", "Demandes_5adamet": "5adamet",
    "Demandes_assam": "assam", "Demandes_barnamjk": "barnamjk",
    "Demandes_chabeb": "chabeb", "Demandes_cv": "cv",
    "Demandes_demande": "demande", "Demandes_enti": "enti",
    "Demandes_labes": "labes", "Demandes_lyoum": "lyoum",
    "Demandes_mar7ba": "mar7ba", "Demandes_n3awnek": "n3awnek",
    "Demandes_nekteblk": "nekteblk", "Demandes_non": "non",
    "Demandes_oui": "oui", "Demandes_radio": "radio",
    "Demandes_se7a": "se7a", "Demandes_siye7a": "siye7a",
    "Demandes_t7eb": "t7eb", "Demandes_ta3lim": "ta3lim",
    "Demandes_ta3raf": "ta3raf", "Demandes_ta9ra": "ta9ra",
    "Demandes_telvza": "telvza", "Demandes_tha9afa": "tha9afa",
    "Destinations_baladya": "baladya", "Destinations_banka": "banka",
    "Destinations_bousta": "bousta", "Destinations_dar": "dar",
    "Destinations_ma7kma": "ma7kma", "Destinations_mostawsaf": "mostawsaf",
    "Destinations_sbitar": "sbitar", "Destinations_wzara": "wzara",
    "Famille_3ayla": "3ayla", "Famille_5al-3am": "5al-3am",
    "Famille_5ou": "5ou", "Famille_bent": "bent",
    "Famille_bou": "bou", "Famille_eben": "eben",
    "Famille_jad": "jad", "Famille_jadda": "jadda",
    "Famille_mar2a": "mar2a", "Famille_o5t": "o5t",
    "Famille_om": "om", "Famille_tfol": "tfol",
    "Jours_5mis": "5mis", "Jours_a7ad": "a7ad",
    "Jours_erb3a": "erb3a", "Jours_jom3a": "jom3a",
    "Jours_sebt": "sebt", "Jours_thleth": "thleth",
    "Jours_thnin": "thnin",
    "Transport_car": "car", "Transport_karhba": "karhba",
    "Transport_louage": "louage", "Transport_métro": "métro",
    "Transport_taxi": "taxi", "Transport_train": "train",
}


# ==========================================
# SCHEMAS
# ==========================================
class LandmarksInput(BaseModel):
    """
    landmarks: liste de 63 valeurs float
    (21 points × 3 coordonnées x,y,z normalisées)
    """
    landmarks: list[float]

class PredictionOutput(BaseModel):
    label_raw: str        # ex: "Demandes_mar7ba"
    arabizi:   str        # ex: "mar7ba"
    arabic:    str        # ex: "مرحبا"
    confidence: float     # probabilité entre 0 et 1
    category:  str        # ex: "Demandes"


# ==========================================
# ROUTES
# ==========================================
@app.get("/")
def root():
    return {
        "service": "Traducteur Langue des Signes Tunisien",
        "status": "online",
        "model_loaded": model is not None,
        "classes": len(labels),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: LandmarksInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(data.landmarks) != 63:
        raise HTTPException(
            status_code=422,
            detail=f"63 landmarks attendus, reçu: {len(data.landmarks)}"
        )

    X = np.array(data.landmarks).reshape(1, -1)

    # Prédiction + probabilités
    label_raw  = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    arabizi = LABEL_TO_ARABIZI.get(label_raw, label_raw)
    arabic  = TRADUCTION_ARABE.get(arabizi, arabizi)
    category = label_raw.split("_")[0] if "_" in label_raw else "Unknown"

    return PredictionOutput(
        label_raw=label_raw,
        arabizi=arabizi,
        arabic=arabic,
        confidence=confidence,
        category=category,
    )

@app.get("/labels")
def get_labels():
    """Retourne toutes les classes du modèle"""
    return {"labels": labels, "count": len(labels)}
