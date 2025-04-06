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
hist -o -p -f classify_name_nationality_heatmap.hist.md
text = "Khalid"
predicted_classes = []
hidden_tensors = []
for i in range(1, len(text) + 1):
    hidden_tensors.append(model.predict_hidden(text[:i]))
    predicted_classes.append(model.predict_category(text[:i]))
df_hidden
df_hidden.corr()['position'][:128].argmax()
df_hidden.corr()['position'][:128].argmin)
df_hidden.corr()['position'][:128].argmin()
df_hidden.corr()['position'].iloc[[11,18]]
hist -o -p
categories
predicted_categories
predicted_classes
pred_categories = predicted_classes
input_texts = [text[:i]) for i in range(1, len(text) + 1)]
input_texts = [text[:i] for i in range(1, len(text) + 1)]
input_texts
pd.Series(index=input_texts, data=pred_categories)
pd.Series(pred_categories, input_texts)
hist
hist -o
df_hidden
df_hidden[:128]
df_hidden[list(range(128))]
hist -o
df_hidden.index = list(text)
df_hidden
df_hidden[list(range(128))]
pd.options.display.max_columns = 12
df_hidden[list(range(128))]
pd.options.display.float_format = '{:.2f}'.format
df_hidden_raw = pd.DataFrame(hidden_lists, index=list(text)).T
df_hidden_raw
df_hidden_raw = pd.DataFrame(hidden_lists, index=list(text))
df_hidden_raw
pd.options.display.float_format = '{: .2f}'.format
df_hidden_raw
pd.options.display.float_format = '{:0.2f}'.format
df_hidden_raw
df_hidden_raw
df_hidden_raw.corr(df_hidden['position'])
df_hidden_raw.corr?
df_hidden_raw.corrwith(df_hidden['position'])
df_hidden_raw.corrwith(range(len(text)))
df_hidden_raw.corrwith(pd.Series(range(len(text))))
df_hidden_raw.corrwith(pd.Series(range(len(text)), index=df_hidden_raw.index))
position = pd.Series(range(len(text)), index=df_hidden_raw.index)
position
pd.DataFrame(position)
pd.DataFrame(position).T
df_hidden_raw.corrwith(position).argmax()
df_hidden_raw.corrwith(position).sort_values()
df_hidden_raw.corrwith(len(text) - position).sort_values()
hist -o -p -f classify_name_nationality_position_correlation.hist.md
hist -f classify_name_nationality_position_correlation.hist.py
