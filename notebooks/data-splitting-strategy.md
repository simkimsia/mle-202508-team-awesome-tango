# Data Splitting Strategy for N-CMAPSS ML Model

This document describes the data splitting strategy used in the N-CMAPSS ML Model notebook.

## Overview

The notebook implements a **hierarchical splitting strategy** that operates at two levels:

1. **Dataset-level split**: Separating OOT (Out-of-Time) test sets from internal datasets
2. **Unit-level split**: Splitting internal datasets into train/validation/test sets by engine units

## Datasets Used

The analysis uses **all 7 N-CMAPSS datasets**:

| Dataset | Filename | Number of Engines |
|---------|----------|-------------------|
| DS01 | N-CMAPSS_DS01.h5 | 10 |
| DS02 | N-CMAPSS_DS02.h5 | 9 |
| DS03 | N-CMAPSS_DS03.h5 | 15 |
| DS04 | N-CMAPSS_DS04.h5 | 10 |
| DS05 | N-CMAPSS_DS05.h5 | 10 |
| DS06 | N-CMAPSS_DS06.h5 | 10 |
| DS07 | N-CMAPSS_DS07.h5 | 10 |
| **Total** | | **74 engines** |

## Data Split Configuration

### Level 1: Dataset-Level Split

```python
TRAIN_SETS = [1, 3, 4, 5, 6]  # Internal datasets
OOT_SETS = [2, 7]              # Out-of-Time test datasets
```

- **Internal datasets** (1, 3, 4, 5, 6): Used for training, validation, and internal testing
- **OOT datasets** (2, 7): Reserved for out-of-time testing to evaluate model generalization

### Level 2: Unit-Level Split

Internal datasets are split by complete engine units (not by individual time steps):

1. **First split** (Internal → Train+Val vs Test):
   - Test: 15% of internal units
   - Train+Val: 85% of internal units

2. **Second split** (Train+Val → Train vs Val):
   - Validation: 15% of train+val units
   - Training: 85% of train+val units

## Split Distribution

### By Engine Count

| Split | Number of Engines | Percentage of Total | Statistical Adequacy |
|-------|-------------------|---------------------|----------------------|
| Training | 39 | 52.7% | ✓ SUFFICIENT (n ≥ 30) |
| Validation | 7 | 9.5% | ✗ INSUFFICIENT |
| Test | 9 | 12.2% | ✗ INSUFFICIENT |
| OOT | 19 | 25.7% | ✗ INSUFFICIENT |
| **Total** | **74** | **100%** | |

### By Row Count

| Split | Number of Rows | Percentage |
|-------|----------------|------------|
| Training | 26,898,226 | 49.0% |
| Validation | 6,905,795 | 12.6% |
| Test | 7,333,005 | 13.4% |
| OOT | 13,737,152 | 25.0% |
| **Total** | **54,874,178** | **100%** |

## Splitting Implementation

### Key Function

The splitting is performed by the `split_data_by_unit()` function:

```python
def split_data_by_unit(df, train_sets, oot_sets):
    """Split data into train, validation, internal test, and out-of-time sets by engine unit."""
    # Step 1: Separate OOT data
    df_oot = df[df['dataset'].isin(oot_sets)].copy()
    df_internal = df[df['dataset'].isin(train_sets)].copy()

    # Step 2: Get unique internal units
    internal_units = df_internal['unit'].unique()

    # Step 3: First split - create test set (15% of internal)
    train_val_units, test_units = train_test_split(
        internal_units,
        test_size=0.15,
        random_state=42
    )

    # Step 4: Create test dataframe
    df_test = df_internal[df_internal['unit'].isin(test_units)].copy()
    df_train_val = df_internal[df_internal['unit'].isin(train_val_units)].copy()

    # Step 5: Second split - create train and validation sets (15% of train_val)
    train_val_units_shuffled = df_train_val['unit'].unique()
    train_units, val_units = train_test_split(
        train_val_units_shuffled,
        test_size=0.15,
        random_state=42
    )

    # Step 6: Create train and validation dataframes
    df_val = df_train_val[df_train_val['unit'].isin(val_units)].copy()
    df_train = df_train_val[df_train_val['unit'].isin(train_units)].copy()

    # Step 7: Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_oot = df_oot.reset_index(drop=True)

    return df_train, df_val, df_test, df_oot
```

### Function Call

```python
df_train, df_val, df_test, df_oot = split_data_by_unit(gold_df, TRAIN_SETS, OOT_SETS)
```

## Splitting Process Sequence

1. **Load all 7 datasets** into a combined dataframe (`gold_df`)
2. **Separate OOT data**: Extract datasets 2 & 7 → `df_oot`
3. **Keep internal data**: Keep datasets 1, 3, 4, 5, 6 → `df_internal`
4. **First split**: Split internal units 85/15 → `train_val_units` / `test_units`
5. **Create test set**: Filter `df_internal` by `test_units` → `df_test`
6. **Second split**: Split train_val units 85/15 → `train_units` / `val_units`
7. **Create validation set**: Filter by `val_units` → `df_val`
8. **Create training set**: Filter by `train_units` → `df_train`

## Key Characteristics

### Splitting Criteria

1. **Unit-Based Splitting**: The split is done by complete engine units, not by individual time steps or rows
   - This prevents data leakage between sets
   - Each engine's complete lifecycle is kept in one split

2. **Random Seed**: Uses `random_state=42` for reproducibility
   - Ensures consistent splits across runs

3. **Dataset-Level OOT**: Entire datasets (2 & 7) are held out as OOT
   - Tests generalization across different operating conditions
   - Simulates real-world deployment scenarios

4. **Hierarchical Splitting**:
   - Test set is created first from internal data
   - Then train/val split is done on the remaining units
   - Maintains proper separation between evaluation sets

## Purpose of Each Split

### Training Set (39 engines, ~27M rows)

- Used for model training and parameter learning
- Statistically sufficient sample size (n ≥ 30)
- Contains majority of data for robust model learning

### Validation Set (7 engines, ~7M rows)

- Used for hyperparameter tuning and model selection
- Prevents overfitting during training
- Note: Small number of engines may limit statistical reliability

### Test Set (9 engines, ~7M rows)

- Used for internal model evaluation
- Tests model performance on unseen data from known conditions
- Note: Small number of engines may limit statistical reliability

### OOT Set (19 engines, ~14M rows)

- Out-of-time test set for evaluating model generalization
- Tests model robustness to different operating conditions
- Simulates real-world deployment on new engine types/conditions
- Note: Limited number of engines for comprehensive OOT validation

## Statistical Considerations

According to the notebook's analysis, only the training set meets the Central Limit Theorem requirement (n ≥ 30) for statistical sufficiency. The validation, test, and OOT sets have fewer than 30 engines, which may limit the statistical reliability of performance metrics calculated on these sets.

However, the large number of rows in each set (millions of time steps) provides substantial data for evaluation, even though the number of independent engine units is limited.

## References

- Notebook file: `notebooks/N CMAPSS ML Model.ipynb`
- Splitting function: Lines 1917-1939
- Function call: Line 1942
