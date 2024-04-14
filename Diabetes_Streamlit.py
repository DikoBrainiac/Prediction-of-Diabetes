# Model metrics
    y_pred = ensemble_rf_et.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    st.write('Model Metrics:')
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')
    st.write(f'AUC: {auc:.2f}')

    # Perform cross-validation with the ensemble classifier
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
    for train_index, test_index in skf.split(X_test_scaled, y_test):
        X_train_fold, X_test_fold = X_test_scaled[train_index], X_test_scaled[test_index]
        y_train_fold, y_test_fold = y_test[train_index], y_test[test_index]
        ensemble_rf_et.fit(X_train_fold, y_train_fold)
        y_pred_fold = ensemble_rf_et.predict(X_test_fold)
        for metric in scores:
            score = None
            if metric == 'accuracy':
                score = accuracy_score(y_test_fold, y_pred_fold)
            elif metric == 'precision':
                score = precision_score(y_test_fold, y_pred_fold)
            elif metric == 'recall':
                score = recall_score(y_test_fold, y_pred_fold)
            elif metric == 'f1':
                score = f1_score(y_test_fold, y_pred_fold)
            elif metric == 'roc_auc':
                score = roc_auc_score(y_test_fold, y_pred_fold)
            scores[metric].append(score)

    # Compute and print mean scores
    mean_scores = {metric: np.mean(scores[metric]) for metric in scores}
    st.write('Mean Scores:')
    for metric, score in mean_scores.items():
        st.write(f'{metric}: {score:.4f}')
