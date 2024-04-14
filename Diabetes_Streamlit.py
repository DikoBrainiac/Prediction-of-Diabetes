elif section == 'Make Predictions':
    st.header('Make Predictions')
    df = load_data()
    model, _, _, scaler, selected_features = train_model(df)  # Load the model and scaler

    if selected_features is not None:  # Ensure selected_features is not None
        input_data = {}
        label_encoder = LabelEncoder()  # Initialize LabelEncoder for smoking_history
        smoking_levels = ['No Info', 'never', 'former', 'current', 'not current', 'ever']
        for feature in selected_features:
            if feature == 'smoking_history':
                selected_smoking_level = st.selectbox(f'Select {feature}', smoking_levels)
                input_data[feature] = label_encoder.fit_transform([selected_smoking_level])[0]  # Encode selected smoking level
            else:
                input_data[feature] = st.number_input(f'Enter {feature}', step=0.01)
        
        if st.button('Predict'):
            # Prepare input data for prediction
            input_features = []
            for feature in selected_features:
                if feature == 'smoking_history':
                    # Convert smoking history to one-hot encoding
                    for level in smoking_levels:
                        if level == input_data['smoking_history']:
                            input_features.append(1)
                        else:
                            input_features.append(0)
                else:
                    input_features.append(input_data[feature])
                    
            input_features = np.array(input_features).reshape(1, -1)
            
            prediction_ensemble = predict_diabetes(model, input_features, scaler)
            st.write(f'Prediction using Ensemble Model (Random Forest + Extra Trees): {prediction_ensemble[0]}')
    else:
        st.warning("Please train the model first before making predictions.")
