# Fact-Checking Evaluation Summary: openai_zero_shot
Generated on: 2025-10-26 14:59:26
============================================================

## Dataset Information
Total samples: 200

True label distribution:
  FALSE: 106 (53.0%)
  MIXED: 34 (17.0%)
  FACT: 60 (30.0%)

Predicted label distribution:
  FALSE: 45 (22.5%)
  MIXED: 147 (73.5%)
  FACT: 8 (4.0%)

## Key Performance Metrics

### Overall Performance
Accuracy: 0.3800
F1-Score (Macro): 0.3506
F1-Score (Micro): 0.3800
F1-Score (Weighted): 0.3868

### Binary Classification (FACT vs non-FACT)
Accuracy: 0.7300
Precision (FACT): 0.8750
Recall (FACT): 0.1167
F1-Score (FACT): 0.2059
Precision (non-FACT): 0.7240
Recall (non-FACT): 0.9929
F1-Score (non-FACT): 0.8373

### AUC Scores
Binary AUC (FACT vs non-FACT): 0.6537
Multiclass AUC (One-vs-Rest): 0.6766

### Per-Class Performance
FALSE:
  Precision: 0.8444
  Recall: 0.3585
  F1-Score: 0.5033

MIXED:
  Precision: 0.2109
  Recall: 0.9118
  F1-Score: 0.3425

FACT:
  Precision: 0.8750
  Recall: 0.1167
  F1-Score: 0.2059

### Confidence Analysis
Mean confidence: 0.7419
Std confidence: 0.1510

Confidence by true label:
  FALSE: 0.7546 ± 0.1738
  MIXED: 0.7276 ± 0.1235
  FACT: 0.7277 ± 0.1157

Confidence by predicted label:
  FALSE: 0.9084 ± 0.0344
  MIXED: 0.6840 ± 0.1336
  FACT: 0.8700 ± 0.0403

### Confusion Matrix
Predicted →
True ↓        FALSE   MIXED    FACT
   FALSE         38      68       0
   MIXED          2      31       1
    FACT          5      48       7

### Detailed Classification Report
```
              precision    recall  f1-score   support

       FALSE       0.84      0.36      0.50       106
       MIXED       0.21      0.91      0.34        34
        FACT       0.88      0.12      0.21        60

    accuracy                           0.38       200
   macro avg       0.64      0.46      0.35       200
weighted avg       0.75      0.38      0.39       200

```