# explainers.py

def show_prediction_history(model, text="Khalid"):
    preds = []
    for i in range(1, len(text) + 1):
        substr = text[:i]
        preds.append([substr, model.predict_category(substr)])
        print(i, substr + ' ' * (len(text) - len(substr)), preds[-1])
    return preds
