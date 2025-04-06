%run ch08_rnn_char_nationality.py
model.predict('Nakamoto')
model.predict_category('Nakamoto')
model.predict_category('Dostoevsky')
model.predict_category("O'Neal")
model.categories
model.predict_category("Smith")
model.predict_category("James")
model.predict_category("Johnson")
model.predict_category("Khalid")
CATEGORIES
model.state_dict()
model.char2i
    category_tensor = torch.tensor([model.categories.index('Arabic')], dtype=torch.long)
    char_seq_tensor = encode_one_hot_seq('Rochdi', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
    category_tensor = torch.tensor([model.categories.index('Arabic')], dtype=torch.long)
    char_seq_tens = encode_one_hot_seq('Rochdi', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
outpu
output
output.numpy().exp()
output.detach().numpy().exp()
np.exp(output.detach().numpy())
model.categories
    category_tensor = torch.tensor([model.categories.index('Brazilian')], dtype=torch.long)
    char_seq_tens = encode_one_hot_seq('James', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
category_tensor
char_seq_tens
model.char2i['J']
char_seq_tens[0][0]
char_seq_tens[0][0].argmax()
output.detach().numpy().exp
np.exp(output.detach().numpy())
np.exp(output.detach().numpy()).argmax()
model.categories[np.exp(output.detach().numpy()).argmax()]
    char_seq_tens = encode_one_hot_seq('Khalid', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
model.categories[np.exp(output.detach().numpy()).argmax()]
    char_seq_tens = encode_one_hot_seq('Khalid', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    outputs = []
    hiddens = []
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
        outputs.append(output)
        hiddens.append(hidden)
cats = []
for v in outputs:
    cats.append(model.categories[np.exp(output.detach().numpy()).argmax()])
cats
    char_seq_tens = encode_one_hot_seq('James', char2i=char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    outputs = []
    hiddens = []
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
        outputs.append(output)
        hiddens.append(hidden)
cats = []
for v in outputs:
    cats.append(model.categories[np.exp(output.detach().numpy()).argmax()])
cats
cats = []
for v in outputs:
    cats.append(model.categories[np.exp(v.detach().numpy()).argmax()])
hist
def visualize_outputs(model, text):
    char_seq_tens = encode_one_hot_seq(text, char2i=model.char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    outputs = []
    hiddens = []
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
        outputs.append(output)
        hiddens.append(hidden)
    cats = []
    for v in outputs:
        cats.append(model.categories[np.exp(v.detach().numpy()).argmax()])
    return cats
visualize_outputs(model, 'Khalid')
visualize_outputs(model, 'Kho')
visualize_outputs(model, 'James')
model.categories
model.categories[2]
df.nationality == model.categories[2]
mask = df.nationality == model.categories[2]
df[mask]
df[mask]['surname']
visualize_outputs(model, 'Silva')
visualize_outputs(model, 'da Silva')
visualize_outputs(model, 'd')
visualize_outputs(model, 'dos')
mask = df.nationality == "Ethiopian"
df[mask]['surname']
visualize_outputs(model, 'Seyoum')
model.categories
visualize_outputs(model, 'Finnish')
df[df.nationality == "Finnish"]['surname']
visualize_outputs(model, 'Virtanen')
visualize_outputs(model, 'Nieminen')
visualize_outputs(model, 'nieminen')
visualize_outputs(model, 'Ibe')
visualize_outputs(model, 'James')
visualize_outputs(model, 'Kho')
visualize_outputs(model, 'Khe')
visualize_outputs(model, 'Chi')
visualize_outputs(model, 'Che')
visualize_outputs(model, 'Ist')
visualize_outputs(model, 'ABC')
hidden
hist -o -p -f ch08_mobprog_surname_nationality_visualization.md
hist -f ch08_mobprog_surname_nationality_visualization.py
