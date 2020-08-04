class Dataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, d_model, num_heads, max_len = 96):
    self.df = df
    self.max_len=max_len
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = self.d_model // self.num_heads
    self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased',
                                                  do_lower_case=True)
    self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
    self.fc1 = nn.Linear(self.d_model, self.d_k)
    self.fc2 = nn.Linear(self.d_model, self.d_k)
    self.fc3 = nn.Linear(self.d_model, self.d_k)
    

  def __len__(self):
    
    return len(self.df)

  def __getitem__(self, idx):
    Length = len(train_df[idx].split())
    
    Query = []
    Key = []
    Value = []
    
    Output = KFMDataset(train_df, self.tokenizer, max_len = 96)
      
    input_ids = Output[idx][0].unsqueeze(0)
    token_type_ids = Output[idx][1].unsqueeze(0)
    attention_mask = Output[idx][2].unsqueeze(0)
    
    embedding = KFM_BERT_Embedding(input_ids, token_type_ids, attention_mask)
    embedding = torch.Tensor(embedding).squeeze(0)
    for i in range(Length):
      
      query = self.fc1(embedding[i])
      Query.append(query)
      key = self.fc2(embedding[i])
      Key.append(key)
      value = self.fc3(embedding[i])
      Value.append(value)

    return (Query, Key, Value)
 
 class Real_Dataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_len = 96):
    self.df = df
    self.max_len = max_len
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                   do_lower_case=True)
  def __len__(self):
    return len(self.df)
 
  def __getitem__(self, idx):
    DF = []
    for i in range(len(train_df[idx].split())):
      DF.append(train_df[idx].split()[i])
    if len(DF) == 0:
      DF = ['i', 'you']
    return DF
 
 class Real_Token_Dataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_len = 96):
    self.df = df
    self.max_len = max_len
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                   do_lower_case=True)
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    DF = []
    Real_Data = self.tokenizer.tokenize(self.df[idx])
    for i in range(len(Real_Data)):
      DF.append(Real_Data[i])
    if len(DF) == 0:
      DF = ['to', 'of']
    return DF
