# ESERISIA AI

Plateforme logicielle Python orientée expérimentation IA (API, interface web, modules d'inférence et d'orchestration).

## Objectif du dépôt

Ce dépôt sert de base technique pour construire et tester des composants IA en environnement local.
Les performances, coûts et niveaux de qualité dépendent du matériel, des dépendances installées et des configurations runtime.

## Composants principaux

- `api/main.py` : API FastAPI.
- `web/app.py` : interface Streamlit.
- `eserisia/inference/` : logique d'inférence.
- `eserisia/ultimate/ultimate_system.py` : orchestrateur multi-modules défensif.
- `tests/` : tests de contrat/smoke.

## Prérequis

- Python 3.11+
- Environnement virtuel recommandé

## Installation

```bash
git clone https://github.com/eserisia/ai.git
cd ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Lancer l'API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Lancer l'interface web

```bash
streamlit run web/app.py
```

## Configuration environnement

Variables supportées:

- `ESERISIA_ALLOWED_ORIGINS` : origines CORS autorisées (CSV).
- `ESERISIA_API_TOKENS` : tokens Bearer autorisés (CSV).
- `ESERISIA_STRICT_MODE` : `1` (défaut) pour réponses API neutres.
- `ESERISIA_ENABLE_AI_CORE` : `1` pour activer le chargement du module AI core dans l'orchestrateur ultimate.

## Tests

```bash
pytest -q tests/api_contract_test.py
```

## Bonnes pratiques

- Ne pas publier de benchmarks comparatifs sans protocole reproductible.
- Documenter les conditions d'exécution (CPU/GPU, versions, charge, dataset) pour toute mesure chiffrée.
- Conserver un mode strict neutre pour les environnements de production.
