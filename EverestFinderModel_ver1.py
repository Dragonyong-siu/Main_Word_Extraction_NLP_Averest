class EverestFinderModel(torch.utils.data.Dataset):
  def __init__(self,df, tokenizer):
    self.tokenizer= BertTokenizer.from_pretrained('bert-large-uncased', 
                                                  do_lower_case=True)
    self.df = df
    self.A = Dataset(train_df, tokenizer, d_model=100, num_heads=4)

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    for idx in range(len(train_df)):
      query = A[idx][0]
      key = A[idx][1]
      
      if len(query) == 0:
        query = torch.Tensor(2, 25)
        key = torch.Tensor(2, 25)
      print(idx)
      print(query)
      query_LIST = []
      for i in range(len(query)):
        word_emb = query[i]
        word_lis = word_emb.tolist()
        query_LIST.append(word_lis) 
  
      query_LIST = torch.Tensor(query_LIST)
      key_LIST = []
      for i in range(len(key)):
        word_emb = key[i]
        word_lis = word_emb.tolist()
        key_LIST.append(word_lis) 
  
      key_LIST = torch.Tensor(key_LIST)
      scores = torch.matmul(query_LIST, key_LIST.transpose(-2, -1))    
      STD = torch.std(scores, -1)
      STD = STD.tolist()
      minimum = min(STD)
      num_std = STD.index(minimum)
      Final = Real_Dataset(train_df, self.tokenizer)[idx][num_std]
      print(Final)
    return Final
