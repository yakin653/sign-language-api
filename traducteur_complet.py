"""
🇹🇳 TRADUCTEUR TUNISIEN - VOIX ARABE gTTS + pygame
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import json
import os
import tempfile
from datetime import datetime
from collections import deque
from gtts import gTTS
import pygame

print("=" * 60)
print("🇹🇳 المترجم التونسي للغة الإشارة")
print("=" * 60)

# Init pygame mixer UNE SEULE FOIS au démarrage
pygame.mixer.init()

# ==========================================
# CONVERSION ARABIZI → ARABE
# ==========================================
def arabizi_to_arabic(text):
    replacements = {
        "2": "أ", "3": "ع", "4": "غ", "5": "خ",
        "6": "ط", "7": "ح", "8": "ق", "9": "ص"
    }
    def convert_word(word):
        return "".join([replacements.get(c, c) for c in word])
    return " ".join([convert_word(w) for w in text.split()])


# ==========================================
# DICTIONNAIRE ARABIZI → ARABE TUNISIEN
# ==========================================
TRADUCTION_ARABE = {
    "3aslema":   "عسلامة",
    "5adamet":   "خدامت",
    "assam":     "اسمع",
    "barnamjk":  "برنامجك",
    "chabeb":    "شباب",
    "cv":        "سي في",
    "demande":   "طلب",
    "enti":      "انتي",
    "labes":     "لاباس",
    "lyoum":     "اليوم",
    "mar7ba":    "مرحبا",
    "n3awnek":   "نعاونك",
    "nekteblk":  "نكتبلك",
    "non":       "لا",
    "oui":       "آه",
    "radio":     "راديو",
    "se7a":      "صحة",
    "siye7a":    "سياحة",
    "t7eb":      "تحب",
    "ta3lim":    "تعليم",
    "ta3raf":    "تعرف",
    "ta9ra":     "تقرا",
    "telvza":    "تلفزة",
    "tha9afa":   "ثقافة",
    "baladya":   "بلدية",
    "banka":     "بنكة",
    "bousta":    "بوسطة",
    "dar":       "دار",
    "ma7kma":    "محكمة",
    "mostawsaf": "مستوصف",
    "sbitar":    "سبيطار",
    "wzara":     "وزارة",
    "3ayla":     "عيلة",
    "5al-3am":   "خال عم",
    "5ou":       "خو",
    "bent":      "بنت",
    "bou":       "بو",
    "eben":      "ابن",
    "jad":       "جد",
    "jadda":     "جدة",
    "mar2a":     "مرأة",
    "o5t":       "أخت",
    "om":        "أم",
    "tfol":      "طفل",
    "5mis":      "خميس",
    "a7ad":      "احد",
    "erb3a":     "اربعاء",
    "jom3a":     "جمعة",
    "sebt":      "سبت",
    "thleth":    "ثلاثاء",
    "thnin":     "اثنين",
    "car":       "كار",
    "karhba":    "كرهبة",
    "louage":    "لواج",
    "métro":     "مترو",
    "taxi":      "تاكسي",
    "train":     "تران",
}


# ==========================================
# FONCTION PARLER EN ARABE
# ==========================================
def parler_arabe(texte_arabe):
    """Prononce le texte en arabe via gTTS + pygame"""
    try:
        print(f"🔊 Prononciation: {texte_arabe}")
        tts = gTTS(text=texte_arabe, lang='ar', slow=False)
        tmp = tempfile.mktemp(suffix='.mp3')
        tts.save(tmp)
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()
        os.remove(tmp)
        return True
    except Exception as e:
        print(f"⚠️ Erreur voix: {e}")
        print("   → Vérifie ta connexion internet (gTTS nécessite internet)")
        return False


class TraducteurTunisien:
    def __init__(self):
        print("\n🚀 Initialisation...")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

        # Charger modèle
        print("   - Chargement du modèle...")
        try:
            model_paths = [
                'modele_tunisien.pkl',
                '../modele_tunisien.pkl',
                'C:/Users/yakin/m3ak/fastapi-server/model/sign_language_traductor/modele_tunisien.pkl'
            ]
            self.model = None
            for path in model_paths:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    print(f"   ✅ Modèle chargé: {path}")
                    break
            if self.model is None:
                print("   ⚠️ Modèle non trouvé - Mode démo")
        except Exception as e:
            print(f"   ⚠️ Erreur: {e}")
            self.model = None

        # Charger labels
        print("   - Chargement des labels...")
        try:
            label_paths = [
                'labels_tunisiens.json',
                'C:/Users/yakin/m3ak/fastapi-server/model/sign_language_traductor/labels_tunisiens.json'
            ]
            self.labels = []
            for path in label_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        self.labels = json.load(f)
                    print(f"   ✅ Labels: {len(self.labels)} mots")
                    break
        except:
            self.labels = []

        self.dico = {
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
            "Transport_taxi": "taxi", "Transport_train": "train"
        }

        # Gestion phrases
        self.phrase = []
        self.phrase_arabe = []
        self.dernier_mot = None
        self.temps_dernier = time.time()
        self.SEUIL_PHRASE = 2.0
        self.historique = deque(maxlen=5)
        self.SEUIL_STABILITE = 3
        self.total_mots = 0
        self.total_phrases = 0

        # Test voix au démarrage
        print("   - Test voix arabe...")
        parler_arabe("مرحبا")

        print("\n✅ المترجم جاهز!")
        if self.model:
            print(f"🧠 Modèle: {len(self.labels)} mots")
        print("👋 أظهر إيدك قدام الكاميرا")
        print("=" * 60)

    def extraire_landmarks(self, hand_landmarks):
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return np.array(landmarks).reshape(1, -1)

    def compter_doigts(self, hand_landmarks):
        doigts = 0
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            doigts += 1
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
            doigts += 1
        if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
            doigts += 1
        if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
            doigts += 1
        if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
            doigts += 1
        return doigts

    def predire(self, landmarks):
        if self.model is None:
            return None
        try:
            prediction = self.model.predict(landmarks)[0]
            if prediction in self.dico:
                return self.dico[prediction]
            return prediction
        except:
            return None

    def stabiliser(self, mot):
        if mot is None:
            return None
        self.historique.append(mot)
        if len(self.historique) == 5:
            from collections import Counter
            compteur = Counter(self.historique)
            mot_freq, occ = compteur.most_common(1)[0]
            if occ >= self.SEUIL_STABILITE:
                return mot_freq
        return None

    def ajouter(self, mot):
        maintenant = time.time()
        if mot != self.dernier_mot:
            self.phrase.append(mot)
            mot_arabe = TRADUCTION_ARABE.get(mot, arabizi_to_arabic(mot))
            self.phrase_arabe.append(mot_arabe)
            self.dernier_mot = mot
            self.temps_dernier = maintenant
            self.total_mots += 1
            print(f"📝 {mot} → {mot_arabe}  |  Phrase: {' '.join(self.phrase)}")
            return True
        return False

    def terminer(self):
        if not self.phrase:
            return None
        maintenant = time.time()
        if (maintenant - self.temps_dernier) > self.SEUIL_PHRASE:
            phrase_affichage = " ".join(self.phrase)
            phrase_arabe = " ".join(self.phrase_arabe)

            print(f"\n📢 Arabizi : {phrase_affichage}")
            print(f"📢 Arabe   : {phrase_arabe}")

            # Parler en arabe
            parler_arabe(phrase_arabe)

            try:
                with open("conversations.txt", "a", encoding='utf-8') as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{ts}] {phrase_affichage} | {phrase_arabe}\n")
            except:
                pass

            self.phrase = []
            self.phrase_arabe = []
            self.dernier_mot = None
            self.total_phrases += 1
            return phrase_arabe
        return None

    def analyser(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        mot_courant = None
        mode_demo = self.model is None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                if mode_demo:
                    doigts = self.compter_doigts(hand_landmarks)
                    mot_courant = f"Doigts: {doigts}"
                    h, w = frame.shape[:2]
                    x = int(hand_landmarks.landmark[0].x * w)
                    y = int(hand_landmarks.landmark[0].y * h) - 50
                    cv2.rectangle(frame, (x-10, y-40), (x+150, y), (0,0,0), -1)
                    cv2.putText(frame, mot_courant, (x, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    landmarks = self.extraire_landmarks(hand_landmarks)
                    mot = self.predire(landmarks)
                    if mot:
                        mot_courant = mot
                        mot_arabe = TRADUCTION_ARABE.get(mot, arabizi_to_arabic(mot))
                        h, w = frame.shape[:2]
                        x = int(hand_landmarks.landmark[0].x * w)
                        y = int(hand_landmarks.landmark[0].y * h) - 60
                        cv2.rectangle(frame, (x-10, y-55), (x+280, y+5), (0,0,0), -1)
                        cv2.putText(frame, mot, (x, y-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        cv2.putText(frame, mot_arabe, (x, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        if not mode_demo and mot_courant:
            mot_stable = self.stabiliser(mot_courant)
            if mot_stable:
                self.ajouter(mot_stable)

        self.terminer()

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w-10, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 85), (0, 255, 100), 2)

        if self.phrase_arabe:
            cv2.putText(frame, " ".join(self.phrase_arabe), (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            msg = "Mode demo..." if mode_demo else "Montre un signe..."
            cv2.putText(frame, msg, (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        return frame


def main():
    traducteur = TraducteurTunisien()

    print("\n📸 Ouverture caméra...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Caméra non trouvée!")
        return

    print("✅ Caméra ouverte! (ESC pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = traducteur.analyser(frame)
        cv2.imshow('Traducteur Langue des Signes Tunisien', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print(f"\n📊 Mots: {traducteur.total_mots} | Phrases: {traducteur.total_phrases}")


if __name__ == "__main__":
    main()