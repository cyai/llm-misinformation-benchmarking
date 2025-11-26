# Fact-Checking Experiment Evaluation Report

Generated: 2025-11-27 02:27:41

## Experiment Configuration

- **Model**: gpt-5-mini-2025-08-07
- **Provider**: openai
- **Strategies**: zero_shot, one_shot, few_shot
- **Test iterations**: 5

## Overall Results

| Strategy | Accuracy | F1-Macro | F1-FACT | F1-FALSE | Samples |
|----------|----------|----------|---------|----------|----------|
| zero_shot | 0.7334 ± 0.0034 | 0.5649 ± 0.0049 | 0.2941 ± 0.0080 | 0.8357 ± 0.0022 | 21155 |
| one_shot | 0.7339 ± 0.0019 | 0.5601 ± 0.0036 | 0.2836 ± 0.0063 | 0.8366 ± 0.0012 | 21155 |
| few_shot | 0.7347 ± 0.0016 | 0.5576 ± 0.0045 | 0.2776 ± 0.0086 | 0.8375 ± 0.0010 | 21155 |

## Detailed Per-Strategy Results

### ZERO_SHOT

**Aggregated Metrics (Mean ± Std):**

- Accuracy: 0.7334 ± 0.0034
- F1-Macro: 0.5649 ± 0.0049
- Precision (FACT): 0.5457 ± 0.0173
- Recall (FACT): 0.2014 ± 0.0058
- F1-Score (FACT): 0.2941 ± 0.0080
- Precision (FALSE): 0.7547 ± 0.0017
- Recall (FALSE): 0.9361 ± 0.0036
- F1-Score (FALSE): 0.8357 ± 0.0022
- Mean Confidence: 0.7949 ± 0.0007
- Total Samples: 21155
- Iterations: 5

**Per-Iteration Results:**

| Iteration | Accuracy | F1-Macro | F1-FACT | F1-FALSE |
|-----------|----------|----------|---------|----------|
| iteration_0 | 0.7315 | 0.5639 | 0.2935 | 0.8343 |
| iteration_1 | 0.7372 | 0.5669 | 0.2953 | 0.8385 |
| iteration_2 | 0.7377 | 0.5722 | 0.3063 | 0.8382 |
| iteration_3 | 0.7291 | 0.5571 | 0.2811 | 0.8331 |
| iteration_4 | 0.7317 | 0.5645 | 0.2946 | 0.8344 |

### ONE_SHOT

**Aggregated Metrics (Mean ± Std):**

- Accuracy: 0.7339 ± 0.0019
- F1-Macro: 0.5601 ± 0.0036
- Precision (FACT): 0.5510 ± 0.0100
- Recall (FACT): 0.1909 ± 0.0049
- F1-Score (FACT): 0.2836 ± 0.0063
- Precision (FALSE): 0.7533 ± 0.0012
- Recall (FALSE): 0.9407 ± 0.0019
- F1-Score (FALSE): 0.8366 ± 0.0012
- Mean Confidence: 0.8357 ± 0.0008
- Total Samples: 21155
- Iterations: 5

**Per-Iteration Results:**

| Iteration | Accuracy | F1-Macro | F1-FACT | F1-FALSE |
|-----------|----------|----------|---------|----------|
| iteration_0 | 0.7336 | 0.5578 | 0.2790 | 0.8366 |
| iteration_1 | 0.7365 | 0.5638 | 0.2894 | 0.8382 |
| iteration_2 | 0.7308 | 0.5562 | 0.2777 | 0.8346 |
| iteration_3 | 0.7353 | 0.5650 | 0.2929 | 0.8372 |
| iteration_4 | 0.7334 | 0.5576 | 0.2788 | 0.8365 |

### FEW_SHOT

**Aggregated Metrics (Mean ± Std):**

- Accuracy: 0.7347 ± 0.0016
- F1-Macro: 0.5576 ± 0.0045
- Precision (FACT): 0.5574 ± 0.0086
- Recall (FACT): 0.1849 ± 0.0073
- F1-Score (FACT): 0.2776 ± 0.0086
- Precision (FALSE): 0.7525 ± 0.0014
- Recall (FALSE): 0.9441 ± 0.0025
- F1-Score (FALSE): 0.8375 ± 0.0010
- Mean Confidence: 0.8283 ± 0.0030
- Total Samples: 21155
- Iterations: 5

**Per-Iteration Results:**

| Iteration | Accuracy | F1-Macro | F1-FACT | F1-FALSE |
|-----------|----------|----------|---------|----------|
| iteration_0 | 0.7351 | 0.5548 | 0.2716 | 0.8381 |
| iteration_1 | 0.7320 | 0.5504 | 0.2646 | 0.8361 |
| iteration_2 | 0.7367 | 0.5592 | 0.2794 | 0.8389 |
| iteration_3 | 0.7343 | 0.5605 | 0.2841 | 0.8369 |
| iteration_4 | 0.7353 | 0.5629 | 0.2884 | 0.8374 |

## Key Findings

### Best Performing Strategy: **zero_shot**

- **Accuracy**: 0.7334 ± 0.0034
- **F1-Macro**: 0.5649 ± 0.0049
- **Consistency**: High (std: 0.0049)

### Recommendations

❌ **Poor performance**: Significant improvements needed.

### Strategy Comparison

- **Performance gap** (best vs worst): 0.0074
- **Finding**: Strategies perform similarly.
