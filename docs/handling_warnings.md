# Common Warnings and How to Handle Them

This document explains common warning messages you may encounter during model training and evaluation.

---

## PyTorch Lightning Warnings

### 1. "predict_dataloader does not have many workers"

**Full Message:**
```
The 'predict_dataloader' does not have many workers which may be a bottleneck. 
Consider increasing the value of the `num_workers` argument to `num_workers=31` 
in the `DataLoader` to improve performance.
```

**What it means:**
- PyTorch Lightning suggests using multiple CPU cores for data loading
- `num_workers=0` (default) uses single-threaded data loading
- `num_workers=31` would use 31 parallel threads

**Should you change it?**

❌ **NO - Not recommended for this use case**

**Reasons:**
- ✓ Current performance is adequate (predictions are fast)
- ✓ Small dataset doesn't benefit from parallel loading
- ✗ More workers = more memory overhead
- ✗ Can cause issues on Windows (multiprocessing complications)
- ✗ Diminishing returns after 4-8 workers

**When to use `num_workers`:**
- Very large datasets (millions of samples)
- Slow data augmentation pipelines
- GPU is idle waiting for data
- Linux/Unix systems (better multiprocessing support)

**How to suppress:**
```python
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
```

---

## Darts Scaler Warnings

### 2. "Only 1 TimeSeries provided, Scaler expected n=39"

**Full Message:**
```
Only 1 TimeSeries (lists) were provided which is lower than the number of 
series (n=39) used to fit Scaler. This can result in a mismatch between 
the series and the underlying transformers.
```

**What it means:**
- The `Scaler` was fitted on 39 engines (training set)
- During prediction, you're transforming 1 engine at a time
- Darts warns about potential shape mismatch

**Should you worry?**

❌ **NO - This is expected and correct behavior**

**Why it's safe:**
1. **Scaler is fitted once on all training data**
   - Learns global statistics (min, max, mean, std)
   - Applied consistently to all engines
   
2. **Prediction happens per-engine**
   - Each engine is predicted independently
   - Scaler transforms each one using the same learned statistics
   
3. **No mismatch occurs**
   - Scaler works element-wise on features
   - Doesn't depend on batch size
   - Same transformation applied to 1 or 39 series

**Example:**
```python
# Training: Fit scaler on 39 engines
scaler.fit([engine1, engine2, ..., engine39])

# Prediction: Transform one at a time (CORRECT)
for engine in test_engines:
    scaled_engine = scaler.transform(engine)  # ✓ Works perfectly
    predictions = model.predict(scaled_engine)
```

**How to suppress:**
```python
import warnings
warnings.filterwarnings('ignore', message='.*lower than the number of series.*')
```

---

## GPU/Hardware Warnings

### 3. "GPU available: True, used: True"

**What it means:**
- PyTorch Lightning reporting hardware detection
- Confirms GPU is available and will be used

**Should you worry?**

✅ **This is informational, not a warning**

**How to suppress:**
```python
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
```

---

## Best Practices for Warning Management

### When to Suppress Warnings

✅ **Safe to suppress:**
- Performance suggestions that don't apply (num_workers)
- Expected behavior warnings (scaler batch size)
- Repetitive hardware detection messages
- Library internal deprecation warnings

✗ **Don't suppress:**
- Data quality warnings (NaN, infinite values)
- Model convergence issues
- Memory warnings
- File not found errors
- Shape mismatch errors (actual errors, not warnings)

### Code Template for Clean Evaluation

```python
import logging
import warnings

# Suppress PyTorch Lightning verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)

# Suppress expected Darts warnings
warnings.filterwarnings('ignore', message='.*lower than the number of series.*')
warnings.filterwarnings('ignore', message='.*does not have many workers.*')

# Your evaluation code here
evaluate_model(...)
```

---

## Understanding Warning Levels

### Logging Levels (from most to least severe):

1. **CRITICAL** - System crash imminent
2. **ERROR** - Something failed, cannot continue
3. **WARNING** - Something unexpected, but can continue
4. **INFO** - Informational messages (GPU available, etc.)
5. **DEBUG** - Detailed diagnostic information

**Setting logging level:**
```python
logging.getLogger("library_name").setLevel(logging.ERROR)
# Only shows ERROR and CRITICAL, suppresses WARNING/INFO/DEBUG
```

---

## Notebook Organization

### Where to Add Warning Suppression

**Option 1: At the start of evaluation cells**
```python
# Cell: Model Evaluation
import logging
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='...')

# Then evaluation code...
```

**Option 2: In a dedicated setup cell**
```python
# Cell: Evaluation Configuration
import logging
import warnings

def suppress_evaluation_warnings():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', message='.*lower than the number of series.*')
    warnings.filterwarnings('ignore', message='.*does not have many workers.*')

suppress_evaluation_warnings()
```

---

## Troubleshooting

### "I suppressed warnings but they still appear"

**Solutions:**

1. **Check warning source:**
   ```python
   import warnings
   warnings.simplefilter('always')  # Show all warnings with source
   # Run your code
   warnings.simplefilter('ignore')  # Turn back off
   ```

2. **Suppress all warnings (nuclear option):**
   ```python
   import warnings
   warnings.filterwarnings('ignore')
   ```

3. **Restart kernel:**
   - Sometimes logging configuration persists
   - Restart → Rerun suppression code

4. **Check if warning is from different library:**
   ```python
   # May need to suppress multiple sources
   logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
   logging.getLogger("darts").setLevel(logging.ERROR)
   logging.getLogger("torch").setLevel(logging.ERROR)
   ```

---

## Summary

| Warning | Safe to Ignore? | How to Fix |
|---------|----------------|------------|
| `num_workers` suggestion | ✅ Yes | Add logging suppression |
| Scaler batch size mismatch | ✅ Yes (expected) | Add warnings filter |
| GPU available messages | ✅ Yes (informational) | Add logging suppression |
| NaN in data | ❌ No | Fix data preprocessing |
| Model not fitted | ❌ No | Load checkpoint file |
| Out of memory | ❌ No | Reduce batch size |

**Golden Rule:** If the code runs successfully and produces correct results, informational warnings can be safely suppressed for cleaner output.

