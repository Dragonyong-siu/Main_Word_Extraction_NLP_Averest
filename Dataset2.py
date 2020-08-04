class Real_Dataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_len = 96):
    self.df = df
    self.max_len = max_len
    self.W2V= word2vec_model
    self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased',
                                                   do_lower_case=True)
  def __len__(self):
    return len(self.df)
 
  def __getitem__(self, idx):
    DF = []
    for i in range(len(train_df[idx].split())):
      if train_df[idx].split()[i] in word_vectors.vocab:
        DF.append(train_df[idx].split()[i])
    if len(DF) == 0:
      DF = ['i', 'you']    
    return DF
