#############################################################################
                        ##K2_Finder_Model##
#############################################################################
class K2_FinderModel(torch.utils.data.Dataset):
  def __init__(self,df, tokenizer):
    self.df = df
    self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased', 
                                                  do_lower_case=True)
    self.B = Dataset(train_df, self.tokenizer, d_model=768, num_heads = 64)
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
      query = self.B[idx][0]
      key = self.B[idx][1]
      
      if len(query) == 0:
        query = torch.Tensor(2, 12)
        key = torch.Tensor(2, 12)
  
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
      
      return Final
