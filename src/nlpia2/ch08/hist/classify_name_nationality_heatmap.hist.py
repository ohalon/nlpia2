%run classify_name_nationality.py
state_dict = model.state_dict()
state_dict
model.predict_category("Khalid")
model.predict_hidden("Khalid")
text = "Khalid"
predicted_classes = []
hidden_tensors = []
for i in range(1, len(text) + 1):
    hidden_tensors.append(model.predict_hidden(text[:i]))
    predicted_classes.append(model.predict_category(text[:i]))
[h[0] for h in hidden_tensors]
h
h = hidden_tensors[0]
h.tolist()
h = hidden_tensors[0].tolist()
h
hidden_lists = [h[0].tolist() for h in hidden_tensors]
hidden_lists
predicted_classes
pd.DataFrame(hidden_lists, index=list(text)).T
df_hidden = pd.DataFrame(hidden_lists, index=list(text)).T
df_hidden.iloc[128] = predicted_classes
predicted_classes = pd.Series(predicted_classes, index=list(text))
predicted_classes
df_hidden.iloc[128] = predicted_classes
pd.DataFrame(predicted_categories)
df_hidden
predicted_classes
predicted_classes..T
predicted_classes.T
df_hidden.T
df_hidden = df_hidden.T
df_hidden['prediction'] = predicted_classes
df_hidden
df_hidden['position'] = range(len(text))
df_hidden['textlen'] = len(text)
df_hidden
df_hidden['pred_index'] = [categories.index(p) for p in df_hidden['prediction']]
df_hidden
df_hidden.index = list(zip(df_hidden.index, df_hidden['prediction']))
df_hidden
df_hidden.drop(columns=['prediction'])
df_hidden = df_hidden.drop(columns=['prediction']).round(2)
df_hidden.corr()
df_hidden.std(axis=0)
df_hidden.std(axis=0).argmin()
df_hidden.std(axis=0).values[:128].argmin()
df_hidden.std(axis=0)[52]
df_hidden.std(axis=0)[51]
df_hidden.std(axis=0)[53]
df_hidden.drop(columns=['textlen'])
hist -o -p -f classify_name_nationality_heatmap.hist.md
hist -f classify_name_nationality_heatmap.hist.py
df_hidden.corr()['position'].argmax()
df_hidden.corr()['position']
df_hidden.corr()['position'][128]
df_hidden.corr()['position'].ilog[128]
df_hidden.corr()['position'].iloc[128]
df_hidden.corr()['position'][:128].argmax
df_hidden.corr()['position'][:128].argmax()
df_hidden.corr()['position'].ilog[18]
df_hidden.corr()['position'].iloc[18]
hist
hist -f classify_name_nationality_heatmap.hist.py
