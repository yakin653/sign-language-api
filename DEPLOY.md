# 🚀 Déploiement - Traducteur LST sur Render

## Structure du dossier à pusher sur GitHub

```
sign_language_traductor/
├── main.py
├── requirements.txt
├── render.yaml
├── labels_tunisiens.json
└── modele_tunisien.pkl     ← géré via Git LFS
```

---

## Étape 1 — Installer Git LFS (pour le .pkl)

```bash
# Windows : télécharger sur https://git-lfs.com
# Puis dans ton terminal :
git lfs install
```

---

## Étape 2 — Créer le repo GitHub

```bash
cd C:\Users\yakin\m3ak\fastapi-server\model\sign_language_traductor

# Copie les nouveaux fichiers (main.py, requirements.txt, render.yaml) ici
# Puis :

git init
git lfs track "*.pkl"
git add .gitattributes
git add .
git commit -m "feat: API traducteur LST"

# Crée un repo sur github.com, puis :
git remote add origin https://github.com/TON_USERNAME/sign-language-api.git
git push -u origin main
```

---

## Étape 3 — Déployer sur Render

1. Va sur https://render.com → **New Web Service**
2. Connecte ton repo GitHub
3. Render détecte automatiquement le `render.yaml`
4. Clique **Deploy**
5. Attends ~3-5 minutes (premier déploiement)
6. Ton API sera disponible sur :
   `https://sign-language-traductor.onrender.com`

---

## Étape 4 — Tester l'API

```bash
# Health check
curl https://sign-language-traductor.onrender.com/health

# Test predict (63 valeurs)
curl -X POST https://sign-language-traductor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [0.0, 0.0, 0.0, ...]}'   # 63 floats
```

---

## Étape 5 — Appel depuis Flutter

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictSign(List<double> landmarks) async {
  final url = Uri.parse('https://sign-language-traductor.onrender.com/predict');
  
  final response = await http.post(
    url,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'landmarks': landmarks}),
  );

  if (response.statusCode == 200) {
    return jsonDecode(response.body);
    // Retourne: { label_raw, arabizi, arabic, confidence, category }
  } else {
    throw Exception('Erreur API: ${response.statusCode}');
  }
}
```

---

## ⚠️ Notes importantes

- **Free tier Render** : le service s'endort après 15 min d'inactivité.
  Premier appel = ~30s de latence (cold start). Upgrade vers Starter ($7/mois) pour éviter ça.
- **MediaPipe côté Flutter** : tu peux extraire les landmarks directement dans Flutter
  avec `google_mlkit_pose_detection` ou `camera` + envoi à l'API.
- **Format landmarks** : 21 points × 3 (x,y,z) = 63 floats, normalisés par rapport au poignet (point 0).
