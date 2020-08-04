class Dataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, d_model, num_heads, max_len=96):
    self.df = df
    self.max_len=max_len
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = self.d_model // self.num_heads
    self.tokenizer= BertTokenizer.from_pretrained('bert-large-uncased',
                                                  do_lower_case=True)
    self.W2V= word2vec_model
    self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
    self.fc1 = nn.Linear(self.d_model, self.d_k)
    self.fc2 = nn.Linear(self.d_model, self.d_k)
    self.fc3 = nn.Linear(self.d_model, self.d_k)
  
  def __len__(self):
    
    return len(self.df)

  def __getitem__(self, idx):
    for i in range(len(train_df[idx].split())):
      train_word=train_df[idx].split()[i]
    inputs = self.tokenizer.encode(train_df[idx])
    tokens = self.tokenizer.tokenize(train_df[idx])  
    tokens = str(tokens)
    tokens = " ".join(tokens.split())
    ids = inputs
    padding_length = self.max_len - len(ids)
    Length = len(train_df[idx].split())
    Query = []
    Key = []
    Value = []
    for i in range(Length):
      if train_df[idx].split()[i] in word_vectors.vocab:
        embedding = self.W2V[train_df[idx].split()[i]]
        embedding = torch.Tensor(embedding)  
        query = self.fc1(embedding)#.view(100, 25)
        Query.append(query)
        key = self.fc2(embedding)#.view(100, 25)
        Key.append(key)
        value = self.fc3(embedding)#.view(100, 25)
        Value.append(value)
    return (Query, Key, Value)
