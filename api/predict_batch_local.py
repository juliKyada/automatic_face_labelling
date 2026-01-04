#!/usr/bin/env python3
"""
Batch prediction script for local development.
Reads a JSON file containing an `items` array with base64 images and optional labels.
"""
import sys
import json
import os
import base64
import statistics
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import preprocess_image, preprocess_for_ethnicity, preprocess_for_emotion  # noqa: E402

MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'labeling_age_gender')

_age_gender_model = None
_ethnicity_model = None
_emotion_model = None


def get_age_gender_model():
    global _age_gender_model
    if _age_gender_model is None:
        path = os.path.join(MODEL_BASE_PATH, 'age_gender_pseudolabel.h5')
        if os.path.exists(path):
            _age_gender_model = __import__('tensorflow.keras.models', fromlist=['load_model']).load_model(  # type: ignore
                path,
                compile=False,
            )
            _age_gender_model.compile(  # type: ignore
                optimizer='adam',
                loss={'age_out': 'mse', 'sex_out': 'binary_crossentropy'},
                metrics={'age_out': 'mae', 'sex_out': 'accuracy'},
            )
    return _age_gender_model


def get_ethnicity_model():
    global _ethnicity_model
    if _ethnicity_model is None:
        candidates = [
            os.path.join(MODEL_BASE_PATH, 'Ethnicity_lebelling.h5'),
            os.path.join(MODEL_BASE_PATH, 'ethnicity_labelling.h5'),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    _ethnicity_model = __import__('tensorflow.keras.models', fromlist=['load_model']).load_model(  # type: ignore
                        path,
                        compile=False,
                    )
                    break
                except Exception:
                    continue
    return _ethnicity_model


def get_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        candidates = [
            os.path.join(MODEL_BASE_PATH, 'emotion_model.h5'),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    _emotion_model = __import__('tensorflow.keras.models', fromlist=['load_model']).load_model(  # type: ignore
                        path,
                        compile=False,
                    )
                    break
                except Exception:
                    continue
    return _emotion_model


def _decode_image(image_str: str) -> bytes:
    raw = image_str or ''
    if ',' in raw:
        raw = raw.split(',')[1]
    return base64.b64decode(raw)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _predict_single(idx: int, entry: Dict[str, Any]) -> Dict[str, Any]:
    image_data = _decode_image(entry.get('image', ''))
    filename = entry.get('filename') or f'item_{idx}'
    true_age = _safe_int(entry.get('true_age'))
    true_gender = _safe_int(entry.get('true_gender'))

    age_gender_model = get_age_gender_model()
    if age_gender_model is None:
        raise RuntimeError('Age/Gender model not found')

    img_input = preprocess_image(image_data)
    if img_input is None:
        raise RuntimeError(f'Failed to preprocess image: {filename}')

    gender_pred, age_pred = age_gender_model.predict(img_input, verbose=0)
    gender_prob = float(gender_pred[0][0])
    gender_label = 'Male' if gender_prob < 0.5 else 'Female'
    gender_confidence = max(gender_prob, 1 - gender_prob)
    age_value = int(round(age_pred[0][0]))

    ethnicity_model = get_ethnicity_model()
    ethnicity_result: Optional[Dict[str, Any]] = None
    if ethnicity_model is not None:
        try:
            eth_input = preprocess_for_ethnicity(image_data)
            if eth_input is not None:
                eth_probs = ethnicity_model.predict(eth_input, verbose=0)
                eth_probs = np.squeeze(eth_probs)
                if eth_probs.ndim == 0:
                    eth_probs = np.array([1.0 - float(eth_probs), float(eth_probs)])
                eth_index = int(np.argmax(eth_probs))
                ethnicity_result = {
                    'label': f'Class_{eth_index}',
                    'confidence': float(np.max(eth_probs)),
                }
        except Exception:
            ethnicity_result = None

    emotion_model = get_emotion_model()
    emotion_classes = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprise',
        'Neutral',
    ]
    emotion_result: Optional[Dict[str, Any]] = None
    if emotion_model is not None:
        try:
            emo_input = preprocess_for_emotion(image_data)
            if emo_input is not None:
                emo_probs = emotion_model.predict(emo_input, verbose=0)
                emo_probs = np.squeeze(emo_probs)
                if emo_probs.ndim == 0:
                    emo_probs = np.array([1.0 - float(emo_probs), float(emo_probs)])
                emo_index = int(np.argmax(emo_probs))
                label = emotion_classes[emo_index] if emo_index < len(emotion_classes) else f'Emotion_{emo_index}'
                emotion_result = {
                    'label': label,
                    'confidence': float(np.max(emo_probs)),
                }
        except Exception:
            emotion_result = None

    image_type = 'Labeled' if true_age is not None and true_gender in (0, 1) else 'Unlabeled'
    age_error = abs(true_age - age_value) if true_age is not None else None
    gender_correct = None
    if true_gender in (0, 1):
        gender_correct = bool((true_gender == 0 and gender_label == 'Male') or (true_gender == 1 and gender_label == 'Female'))

    return {
        'filename': filename,
        'image_type': image_type,
        'true_age': true_age,
        'true_gender': true_gender,
        'predicted_age': age_value,
        'predicted_gender': gender_label,
        'predicted_nationality': ethnicity_result['label'] if ethnicity_result else None,
        'nationality_confidence': ethnicity_result['confidence'] if ethnicity_result else None,
        'predicted_emotion': emotion_result['label'] if emotion_result else None,
        'emotion_confidence': emotion_result['confidence'] if emotion_result else None,
        'age_error': age_error,
        'gender_correct': gender_correct,
        'gender_confidence': gender_confidence,
        'raw_gender_prob': gender_prob,
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in results if r.get('image_type') == 'Labeled']
    unlabeled = [r for r in results if r.get('image_type') == 'Unlabeled']

    def _mean_safe(values: List[float]) -> Optional[float]:
        clean = [v for v in values if v is not None]
        return statistics.mean(clean) if clean else None

    age_mae = _mean_safe([r.get('age_error') for r in labeled if r.get('age_error') is not None])
    gender_accuracy = _mean_safe([1.0 if r.get('gender_correct') else 0.0 for r in labeled if r.get('gender_correct') is not None])
    avg_confidence = _mean_safe([r.get('gender_confidence') for r in results if r.get('gender_confidence') is not None])

    eth_counts: Dict[str, int] = {}
    emo_counts: Dict[str, int] = {}
    for r in results:
        eth = r.get('predicted_nationality')
        emo = r.get('predicted_emotion')
        if eth:
            eth_counts[eth] = eth_counts.get(eth, 0) + 1
        if emo:
            emo_counts[emo] = emo_counts.get(emo, 0) + 1

    return {
        'total': len(results),
        'labeled_count': len(labeled),
        'unlabeled_count': len(unlabeled),
        'age_mae': age_mae,
        'gender_accuracy': gender_accuracy,
        'avg_confidence': avg_confidence,
        'ethnicity_distribution': eth_counts,
        'emotion_distribution': emo_counts,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)

    input_path = sys.argv[1]
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        print(json.dumps({'success': False, 'error': f'Failed to read input: {exc}'}))
        sys.exit(1)

    items = payload.get('items', [])
    if not isinstance(items, list) or len(items) == 0:
        print(json.dumps({'success': False, 'error': 'No items supplied'}))
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, entry in enumerate(items):
        try:
            result = _predict_single(idx, entry)
            results.append(result)
        except Exception as exc:
            errors.append(f"{entry.get('filename', f'item_{idx}')}: {exc}")
            continue

    summary = _summarize(results)

    print(json.dumps({
        'success': True,
        'results': results,
        'summary': summary,
        'errors': errors,
    }))


if __name__ == '__main__':
    main()
