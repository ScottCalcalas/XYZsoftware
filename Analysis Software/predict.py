import pandas as pd
import joblib

# Load the saved model, scaler, and feature order
log_reg = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = joblib.load('feature_order.pkl')  # Load the saved feature order

# Define numeric and categorical columns
numeric_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
sample_input = {
    'age': 35, 'job': 'technician', 'marital': 'married', 'education': 'university.degree',
    'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'telephone', 'month': 'may',
    'day_of_week': 'mon', 'campaign': 1, 'pdays': 999, 'previous': 0, 'poutcome': 'nonexistent',
    'emp.var.rate': 1.1, 'cons.price.idx': 93.994, 'cons.conf.idx': -36.4, 'euribor3m': 4.857, 'nr.employed': 5191
}

# Define valid categories for categorical variables
valid_categories = {
    'job': ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed',
            'self-employed', 'unknown', 'entrepreneur', 'student'],
    'marital': ['married', 'single', 'divorced', 'unknown'],
    'education': ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 
                  'university.degree', 'illiterate'],
    'default': ['no', 'yes', 'unknown'],
    'housing': ['no', 'yes', 'unknown'],
    'loan': ['no', 'yes', 'unknown'],
    'contact': ['telephone', 'cellular'],
    'month': ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'poutcome': ['nonexistent', 'failure', 'success']
}

def get_user_input():
    """
    Collects user input interactively, replacing invalid inputs with defaults.
    """
    user_input = {}
    print("\nEnter values for the new data. Valid options are provided for each field.")
    for key, example in sample_input.items():
        if key in valid_categories:  # Show valid options for categorical variables
            print(f"\n --------- Valid options for {key}: {valid_categories[key]} --------- ")
        value = input(f"\n{key}: ").strip()
        if key in numeric_cols:  # Handle numeric fields
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                print(f" <<<- Invalid input for {key}. Using default value {example}.")
                value = example
        elif key in valid_categories:  # Validate categorical fields
            if value not in valid_categories[key]:
                print(f" <<<- Invalid input for {key}. Using default value {example}.")
                value = example
        user_input[key] = value if value else example  # Default value if empty
    return user_input

def predict_new_data(new_data):
    """
    Predicts the output for new data, ensuring the feature order matches the training set.
    """
    new_data_df = pd.DataFrame([new_data])

    # Ensure all columns are present
    for col in categorical_cols + numeric_cols:
        if col not in new_data_df.columns:
            new_data_df[col] = sample_input[col]

    # Encode categorical data
    encoded_new_data = pd.get_dummies(new_data_df[categorical_cols])
    encoded_new_df = encoded_new_data.reindex(columns=feature_order[len(numeric_cols):], fill_value=0)

    # Scale numeric data
    numeric_new_data = pd.DataFrame(scaler.transform(new_data_df[numeric_cols]), columns=numeric_cols)

    # Combine numeric and encoded data
    full_new_data = pd.concat([numeric_new_data, encoded_new_df], axis=1)

    # Reorder columns to match the saved feature order
    full_new_data = full_new_data.reindex(columns=feature_order, fill_value=0)

    # Get probabilities for each class
    probabilities = log_reg.predict_proba(full_new_data)[0]
    prediction = 'yes' if probabilities[1] > 0.5 else 'no'

    # Print probabilities
    print(f"\nPredicted probabilities: no = {probabilities[0]:.2f}, yes = {probabilities[1]:.2f}")
    return prediction

if __name__ == "__main__":
    print("Welcome to the Interactive Prediction Program!")
    while True:
        user_data = get_user_input()
        try:
            prediction = predict_new_data(user_data)
            print(f"\nThe predicted outcome is: {prediction}")
        except Exception as e:
            print(f"Error during prediction: {e}")
        cont = input("\nDo you want to input another set of data? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting the program. Thank you!")
            break
