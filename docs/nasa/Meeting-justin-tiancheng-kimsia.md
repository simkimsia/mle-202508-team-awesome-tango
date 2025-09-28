# Issue with data

- The NASA paper it uses a smaller dataset (dataset Kaggle)
- There is a bigger dataset (dataset Original 15 gb)

1. Original has the same columns as the Kaggle and more.
2. Original has the same data rows as the Kaggle and more.

we either use the Original or the Kaggle but not together.

we make the Original to have the same columns as the Kaggle by removing the extra columns in Original

Train - 60%
Validate - 20%
Test - 20%
OOT -

The discussion between Justin, Tiancheng, and Kimsia is :

1. justin to take the original dataset 15gb and make it smaller. pick a smaller set of engines and not vary their lifecycles.
    1. this is so that we have enough data for our chosen LSTM model
    2. the kaggle dataset seems to be far less than 100k may not eb enough for LSTM
    3. also justin pointed out the kaggle dataset jupyter notebook they finally train with the entire dataset which makes their predicition suspociously high
2. we choose LSTM because:
    1. nature of problem is time dependent process - how much RUL left is dependent on past events
    2. using point 1 we narrow down to a set of suitable models: LSTM, GRU, TCN, Transformers,
       1. then we eliminate all the unsuitable due to various reasons such as memory of past dependencies not good enough, too simple, require too much data, etc.
 3. As for the issue of explanability, LSTM as pointed out by Tiancheng, is blackbox model with poor interpretability.
    1. However, this doesn't change the nature of the problem which is this requires temporal modeling.
    2. All temporal modeling choices (GRU, TCN, etc) all suffer from poor interpretability anyway, so try to choose the most suitable one for performance reasons
    3. We also interview SIA pilot, and engineer from ST aerospace. Singapore doesn't really practice predictive maintenance given the age of the engines, but they care more about performance than explanability anyway
    4. We chose the most suitable temporal model (LSTM) and addressed explainability through industry-standard tools like SHAP and LIME.