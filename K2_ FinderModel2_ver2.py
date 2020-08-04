###############################################################################
                      ##K2_ Finder_Model_2##
###############################################################################
class K2_FinderModel2(torch.utils.data.Dataset):
  def __init__(self,df, tokenizer):
    self.df = df
    self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased', 
                                                  do_lower_case=True)
    
    self.features = KFMDataset(self.df, self.tokenizer, max_len = 96)
    
    self.attn = Attention_Weight_BERT
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
      self.input_ids = self.features[idx][0].unsqueeze(0)
      self.token_type_ids = self.features[idx][1].unsqueeze(0)
      self.attention_mask = self.features[idx][2].unsqueeze(0)
      
# soft max를 거친 attention_weights 행렬얻기
      Attn_weights = self.attn(self.input_ids, self.token_type_ids, self.attention_mask)
      Ch_Attn_weights = Attn_weights[7].squeeze(0)[7]
# 위에서 얻은 attention_weights는 max_len = 96에 맞춰진 96 *96 행렬이므로 
# 단어 추출을 위해서는 [CLS], [SEP]까지 고려한 TOKEN의 개수로 행렬을 뽑아야한다.
# ----- [CLS]와 [SEP] 는 제외하기
      
      input_ex = self.df[idx]
      A = len(tokenizer(input_ex)['input_ids'])
      Ch_Attn_weights = Ch_Attn_weights[1:A-1, 1:A-1]
      
      STD = torch.std(Ch_Attn_weights, -1)
      STD = STD.tolist()
  
      minimum = min(STD)
      num_std = STD.index(minimum)
     
      Final = Real_Dataset(train_df, self.tokenizer)[idx][num_std]
      
      return Final
