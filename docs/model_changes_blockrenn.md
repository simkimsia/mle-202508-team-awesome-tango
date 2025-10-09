# Model Architecture Change: RNNModel → BlockRNNModel

## Problem
`RNNModel` in Darts doesn't support `past_covariates` by default - it's designed for autoregressive forecasting where the model only uses past values of the target series itself.

**Error encountered:**
```
ValueError: The model does not support `past_covariates`. It only supports `future_covariates`.
```

## Solution
Switch from `RNNModel` to `BlockRNNModel`, which is specifically designed to support past covariates.

## Key Differences

| Feature | RNNModel | BlockRNNModel |
|---------|----------|---------------|
| **Past Covariates Support** | ❌ No | ✅ Yes |
| **Use Case** | Autoregressive (uses only target history) | Multivariate (uses sensor features) |
| **Architecture** | Encoder-decoder for sequences | Encoder + FC layers for fixed outputs |
| **Our Application** | ❌ Not suitable | ✅ Perfect fit |

## Changes Made

### 1. Import Statement
```python
# OLD
from darts.models import RNNModel, TCNModel

# NEW
from darts.models import BlockRNNModel, TCNModel
```

### 2. LSTM Model Definition
```python
# OLD
lstm_model = RNNModel(
    model='LSTM',
    input_chunk_length=SEQUENCE_LENGTH,
    training_length=SEQUENCE_LENGTH,
    ...
)

# NEW
lstm_model = BlockRNNModel(
    model='LSTM',
    input_chunk_length=SEQUENCE_LENGTH,
    output_chunk_length=1,
    n_rnn_layers=2,        # Number of LSTM layers
    hidden_dim=128,        # Hidden layer size
    ...
)
```

### 3. RNN Model Definition
```python
# OLD
rnn_model = RNNModel(
    model='RNN',
    training_length=SEQUENCE_LENGTH,
    ...
)

# NEW
rnn_model = BlockRNNModel(
    model='RNN',
    input_chunk_length=SEQUENCE_LENGTH,
    output_chunk_length=1,
    n_rnn_layers=2,
    hidden_dim=128,
    ...
)
```

## BlockRNNModel Supported Variants

BlockRNNModel supports three RNN types:
- ✅ **`model='LSTM'`** - Long Short-Term Memory (what we're using)
- ✅ **`model='RNN'`** - Vanilla RNN
- ✅ **`model='GRU'`** - Gated Recurrent Unit

All three variants support `past_covariates` for multivariate time series forecasting.

## Architecture Details

**BlockRNNModel Architecture:**
```
Input: [Target History + Sensor Features (past_covariates)]
    ↓
[RNN Encoder] → processes sequences
    ↓
[Fully Connected Layers] → produces predictions
    ↓
Output: [RUL Prediction]
```

**Parameters Added:**
- `n_rnn_layers=2` - Deeper network for complex patterns
- `hidden_dim=128` - Larger hidden state for richer representations
- Removed `training_length` (not needed in BlockRNNModel)

## Why This Works for RUL Prediction

1. **Multivariate Input**: Uses 12 sensor features + RUL history
2. **Sequential Processing**: LSTM captures temporal degradation patterns
3. **Fixed Output**: Predicts next RUL value (output_chunk_length=1)
4. **Past Covariates**: Sensor readings are only available up to current time (not future)

## Training Flow

```python
lstm_model.fit(
    series=train_targets_scaled,              # RUL values (what to predict)
    past_covariates=train_covariates_scaled,  # Sensor features (12 features)
    val_series=val_targets_scaled,
    val_past_covariates=val_covariates_scaled,
    verbose=True
)
```

The model learns:
- How sensor patterns (covariates) correlate with degradation (RUL decrease)
- Temporal dependencies in both sensors and RUL
- Complex multivariate relationships

## Result

✅ Model now correctly handles multivariate time series forecasting with sensor features as inputs.
