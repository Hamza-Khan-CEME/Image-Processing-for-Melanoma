import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split

# Load CSV file
file_path = 'refined_dataset.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("First few rows of the data")
print(df.head())

print("\nColumn Names in data")
print(df.columns)

print("\nData type")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nBasic Info")
print(df.info())

# Correct path format
base_path = "PH2 Dataset images"

# Map categorical features
mappings = {'T': 1, 'A': 0, 'P': 1, 'AT': 0}
categorical_features = ['Pigment Network', 'Dots/Globules', 'Streaks', 'Regression Areas', 'Blue-Whitish Veil']

for feature in categorical_features:
    if feature in df.columns:
        df[feature] = df[feature].map(mappings).fillna(0)

# Map class labels
class_labels = {0: 'Common Nevi', 1: 'Atypical Nevi', 2: 'Melanoma'}
df['Clinical Diagnosis'] = df['Clinical Diagnosis'].map(class_labels)

# Handle numeric features
features = ['Asymmetry', 'Pigment Network', 'Dots/Globules', 'Streaks',
            'Regression Areas', 'Blue-Whitish Veil', 'White', 'Red',
            'Light-Brown', 'Dark-Brown', 'Blue-Gray', 'Black']

for feature in features:
    if feature in df.columns:
        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)

# Debugging: Ensure correct mapping and features
print("Unique values in 'Clinical Diagnosis':", df['Clinical Diagnosis'].unique())
for feature in features:
    if feature not in df.columns:
        print(f"Feature missing: {feature}")
    elif not pd.api.types.is_numeric_dtype(df[feature]):
        print(f"Feature not numeric: {feature}")

# Boxplots for features
for feature in features:
    if feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Clinical Diagnosis', y=feature, palette="Set3")
        plt.title(f"Distribution of {feature} Across Diagnosis Classes")
        plt.xlabel("Diagnosis Class")
        plt.ylabel(feature)
        plt.show()

# Scatterplots for features
for feature in features:
    if feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=df.index, y=feature, hue='Clinical Diagnosis', palette="Set2", s=50)
        plt.title(f"Scatter Plot of {feature} Across Diagnosis Classes")
        plt.xlabel("Data Point Index")
        plt.ylabel(feature)
        plt.legend(title="Diagnosis Class")
        plt.show()

# Helper function to plot histogram
def plot_histogram(image, title):
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend(['Red', 'Green', 'Blue'])
    plt.tight_layout()

# Generate histograms for each class
classes = df['Clinical Diagnosis'].unique()
print(f"\nClasses in dataset: {classes}")

for diagnosis_class in classes:
    class_images = df[df['Clinical Diagnosis'] == diagnosis_class]['Name'].tolist()
    print(f"\nClass: {diagnosis_class}, Number of images: {len(class_images)}")

    if len(class_images) == 0:
        print(f"No images available for class {diagnosis_class}. Skipping.")
        continue

    selected_images = random.sample(class_images, min(5, len(class_images)))
    print(f"Selected images for {diagnosis_class}: {selected_images}")

    for image_name in selected_images:
        dermoscopic_path = os.path.join(base_path, image_name, f"{image_name}_Dermoscopic_Image", f"{image_name}.bmp")
        print(f"Checking path: {dermoscopic_path}")

        if os.path.exists(dermoscopic_path):
            try:
                image = cv2.cvtColor(cv2.imread(dermoscopic_path), cv2.COLOR_BGR2RGB)
                plot_histogram(image, f"Histogram for {diagnosis_class} - {image_name}")
                plt.show()
            except Exception as e:
                print(f"Error plotting histogram for {image_name}: {e}")
        else:
            print(f"File not found: {dermoscopic_path}")

# Connected Components Analysis (CCA)
def apply_cca(mask):
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_labels):  # Skip the background (label 0)
        output[labels == i] = [random.randint(0, 255) for _ in range(3)]
    return output

# Generate CCA visualizations for lesion masks
for diagnosis_class in classes:
    class_images = df[df['Clinical Diagnosis'] == diagnosis_class]['Name'].tolist()
    selected_images = random.sample(class_images, min(5, len(class_images)))

    plt.figure(figsize=(15, 15))
    plt.suptitle(f"Lesion Masks with CCA - {diagnosis_class}", fontsize=16)

    for idx, image_name in enumerate(selected_images):
        lesion_path = os.path.join(base_path, image_name, f"{image_name}_lesion", f"{image_name}_lesion.bmp")

        if os.path.exists(lesion_path):
            mask = cv2.imread(lesion_path, cv2.IMREAD_GRAYSCALE)
            cca_result = apply_cca(mask)

            plt.subplot(1, 5, idx + 1)
            plt.imshow(cca_result)
            plt.title(f"Image: {image_name}")
            plt.axis('off')
        else:
            print(f"File not found: {lesion_path}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


# Analysis to find optimal thresholds for feature segregation

# Compute descriptive statistics for features by class
analysis_results = {}
for feature in features:
    if feature in df.columns:
        print(f"\nAnalysis for Feature: {feature}")
        class_stats = df.groupby('Clinical Diagnosis')[feature].describe()
        print(class_stats)

        # Identify potential thresholds
        means = class_stats['mean']
        stds = class_stats['std']
        thresholds = {
            'lower_bound': means - stds,
            'upper_bound': means + stds
        }
        print(f"Thresholds for {feature}:")
        print(pd.DataFrame(thresholds))

        # Store results for further use
        analysis_results[feature] = {
            'stats': class_stats,
            'thresholds': thresholds
        }

# Visualize potential thresholds for a selected feature
selected_feature = "Asymmetry"  # Example: Replace with any feature from the list
if selected_feature in analysis_results:
    stats = analysis_results[selected_feature]['stats']
    thresholds = analysis_results[selected_feature]['thresholds']

    # Plot the feature distribution with thresholds
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Clinical Diagnosis', y=selected_feature, palette="Set3")
    for class_name, lower, upper in zip(stats.index, thresholds['lower_bound'], thresholds['upper_bound']):
        plt.axhline(lower, color='r', linestyle='--', label=f'{class_name} Lower Bound')
        plt.axhline(upper, color='g', linestyle='--', label=f'{class_name} Upper Bound')
    plt.title(f"Feature Analysis: {selected_feature}")
    plt.xlabel("Diagnosis Class")
    plt.ylabel(selected_feature)
    plt.legend(loc='upper right')
    plt.show()


# List of features to analyze
features_to_analyze = ['Asymmetry', 'Pigment Network', 'Dots/Globules', 'Streaks',
                        'Regression Areas', 'Blue-Whitish Veil', 'White', 'Red',
                        'Light-Brown', 'Dark-Brown', 'Blue-Gray', 'Black']

# Validate features against DataFrame columns
valid_features = [feature for feature in features_to_analyze if feature in df.columns]
missing_features = [feature for feature in features_to_analyze if feature not in df.columns]

# Log missing features
if missing_features:
    print(f"The following features are missing from the DataFrame and will be skipped: {missing_features}")

# Ensure at least one valid feature is present
if not valid_features:
    raise ValueError("None of the features to analyze are present in the DataFrame.")

# Automate feature analysis and threshold calculation
for selected_feature in valid_features:
    # Debugging: Check which feature is being processed
    print(f"Processing feature: {selected_feature}")

    plt.figure(figsize=(10, 6))
    
    try:
        # Plot boxplot for the selected feature
        sns.boxplot(data=df, x='Clinical Diagnosis', y=selected_feature, palette="Set2")
        
        # Add thresholds (mean ± std) for each class
        for diagnosis_class in df['Clinical Diagnosis'].unique():
            class_data = df[df['Clinical Diagnosis'] == diagnosis_class][selected_feature]
            
            if not class_data.empty:
                mean = class_data.mean()
                std = class_data.std()
                lower_bound = mean - std
                upper_bound = mean + std

                # Plot thresholds
                plt.axhline(lower_bound, color='r', linestyle='--', label=f"{diagnosis_class} Lower Bound")
                plt.axhline(upper_bound, color='g', linestyle='--', label=f"{diagnosis_class} Upper Bound")
        
        # Customize the plot
        plt.title(f"Feature Analysis: {selected_feature}")
        plt.xlabel("Diagnosis Class")
        plt.ylabel(selected_feature)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.show()  # Display the plot directly

    except Exception as e:
        print(f"Error processing feature '{selected_feature}': {e}")


#Code for Accuracy Calculation and Visualization
# Split data into training (80%) and testing (20%)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Clinical Diagnosis'])
print(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")

# Use thresholds from training data
thresholds = {}

# Calculate thresholds using the training data
for feature in features_to_analyze:
    if feature in train_data.columns:
        class_stats = train_data.groupby('Clinical Diagnosis')[feature].describe()
        means = class_stats['mean']
        stds = class_stats['std']
        thresholds[feature] = {
            'lower_bound': means - stds,
            'upper_bound': means + stds,
        }

# Predict classes based on thresholds for the test data
def predict_class(row):
    predictions = []
    for feature in features_to_analyze:
        if feature in thresholds and feature in row:
            for class_name, lower, upper in zip(thresholds[feature]['lower_bound'].index, 
                                                thresholds[feature]['lower_bound'].values, 
                                                thresholds[feature]['upper_bound'].values):
                if lower <= row[feature] <= upper:
                    predictions.append(class_name)
    return random.choice(predictions) if predictions else None

# Apply prediction function
test_data['Predicted Diagnosis'] = test_data.apply(predict_class, axis=1)

# Calculate accuracy
correct_predictions = (test_data['Clinical Diagnosis'] == test_data['Predicted Diagnosis']).sum()
total_predictions = len(test_data)
accuracy = correct_predictions / total_predictions

print(f"\nAccuracy on test data: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

# Plot true vs predicted classes
plt.figure(figsize=(10, 6))
sns.countplot(data=test_data, x='Clinical Diagnosis', hue='Predicted Diagnosis', palette='Set2')
plt.title("True vs Predicted Classes")
plt.xlabel("True Diagnosis Class")
plt.ylabel("Count")
plt.legend(title="Predicted Diagnosis")
plt.show()

# Visualize mismatched predictions
mismatches = test_data[test_data['Clinical Diagnosis'] != test_data['Predicted Diagnosis']]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mismatches, x=mismatches.index, y='Predicted Diagnosis', hue='Clinical Diagnosis', palette='Set3', s=50)
plt.title("Mismatched Predictions in Test Data")
plt.xlabel("Data Point Index")
plt.ylabel("Predicted Diagnosis")
plt.legend(title="True Diagnosis")
plt.show()

