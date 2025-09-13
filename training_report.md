(vertual) PS C:\Users\supri\OneDrive\Desktop\water_quality> python main.py train
🚰 WATER QUALITY PREDICTION PIPELINE
==================================================

📊 STEP 1: Data Loading and Preparation
----------------------------------------
🔄 Starting data loading and preparation pipeline...
✅ Data loaded successfully from data/synthetic_water_quality_dataset.csv
   Shape: (10000, 10)
   Features: ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
   Target: Potability
✅ Features and target prepared
   Features shape: (10000, 9)
   Target distribution: {1: 6500, 0: 3500}
✅ Data split completed
   Training set: 8000 samples
   Testing set: 2000 samples
   Training target distribution: {1: 5200, 0: 2800}
   Testing target distribution: {1: 1300, 0: 700}
✅ Features scaled using StandardScaler
   Feature means (after scaling): {'ph': 0.0, 'Hardness': -0.0, 'Solids': -0.0, 'Chloramines': 0.0, 'Sulfate': -0.0, 'Conductivity': 0.0, 'Organic_carbon': 0.0, 'Trihalomethanes': -0.0, 'Turbidity': 0.0}
   Feature stds (after scaling): {'ph': 1.0, 'Hardness': 1.0, 'Solids': 1.0, 'Chloramines': 1.0, 'Sulfate': 1.0, 'Conductivity': 1.0, 'Organic_carbon': 1.0, 'Trihalomethanes': 1.0, 'Turbidity': 1.0}
✅ Data preparation pipeline completed!

🔧 STEP 2: Feature Engineering
----------------------------------------
🔄 Starting feature engineering pipeline...
✅ Created 4 ratio features
✅ Created 5 interaction features
✅ Feature engineering completed!
   Original features: 9
   Engineered features: 18
🔄 Starting feature engineering pipeline...
✅ Created 4 ratio features
✅ Created 5 interaction features
✅ Feature engineering completed!
   Original features: 9
   Engineered features: 18

🤖 STEP 3: Model Training
----------------------------------------
🔄 Training XGBoost model with visual progress...
📊 Training Configuration:
   Training samples: 8,000
   Features: 18
   Estimators: 200
   Learning rate: 0.1
   Max depth: 6
   Validation samples: 2,000

🚀 Starting training process...
📊 Training in progress...
[0]     validation_0-logloss:0.61970    validation_1-logloss:0.62064
[20]    validation_0-logloss:0.12586    validation_1-logloss:0.14015
[40]    validation_0-logloss:0.04584    validation_1-logloss:0.06161
[60]    validation_0-logloss:0.02242    validation_1-logloss:0.03823
[80]    validation_0-logloss:0.01294    validation_1-logloss:0.02910
[100]   validation_0-logloss:0.00844    validation_1-logloss:0.02479
[120]   validation_0-logloss:0.00601    validation_1-logloss:0.02243
[140]   validation_0-logloss:0.00457    validation_1-logloss:0.02181
[160]   validation_0-logloss:0.00360    validation_1-logloss:0.02149
[180]   validation_0-logloss:0.00300    validation_1-logloss:0.02142
[191]   validation_0-logloss:0.00276    validation_1-logloss:0.02166

⏱️  Simulating training progress visualization...
Training Complete: 100%|██████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 62.89%/s] 
✨ Training visualization complete!
✅ Model training completed!
   Training accuracy: 1.0000
   Validation accuracy: 0.9925
📊 Available training history keys: ['validation_0', 'validation_1']
📈 Training curves saved as 'training_progress.png'

📈 STEP 4: Model Evaluation
----------------------------------------
🔄 Evaluating model performance...
✅ Model evaluation completed!
   Accuracy: 0.9925
   AUC Score: 0.9999
   Precision (Safe): 0.9886
   Recall (Safe): 1.0000
   Precision (Unsafe): 1.0000
   Recall (Unsafe): 0.9786

📊 STEP 5: Feature Importance Analysis
----------------------------------------
Top 10 Most Important Features:
                              feature  importance
Turbidity_Trihalomethanes_interaction    0.153331
                             Hardness    0.112259
                              Sulfate    0.096532
                                   ph    0.096472
                      Trihalomethanes    0.089252
                          Chloramines    0.082535
                         Conductivity    0.074871
                       Organic_carbon    0.065684
                               Solids    0.057195
           ph_Chloramines_interaction    0.044838

✅ STEP 6: Cross-validation
----------------------------------------
🔄 Performing 5-fold cross-validation...
✅ Cross-validation completed!
   Mean accuracy: 0.9947 (+/- 0.0043)

💾 STEP 7: Saving Model
----------------------------------------
✅ Model saved to: models/water_quality_xgboost.pkl

🎉 TRAINING PIPELINE COMPLETED!
==================================================