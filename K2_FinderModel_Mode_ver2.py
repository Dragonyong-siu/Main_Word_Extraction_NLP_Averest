###############################################################################
                          ##K2_Finder_Model_Mode##
###############################################################################
# K2_FM 에서는 QUERY와 KEY 를 뽑아 Attention score 을 계산하고 std 를 이용한다.
# K2_FM_Mode"최빈수 맵"을 작성한다.
# K2_FM_Mode에서는 4번째 Layer 중에 4번째 head를 이용하도록 한다.
from collections import Counter
class K2_FinderModel_Mode(torch.utils.data.Dataset):
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
      Ch_Attn_weights = Attn_weights[3].squeeze(0)[3]
# 위에서 얻은 attention_weights는 max_len = 96에 맞춰진 96 *96 행렬이므로 
# 단어 추출을 위해서는 [CLS], [SEP]까지 고려한 TOKEN의 개수로 행렬을 뽑아야한다.
# ----- [CLS]와 [SEP] 는 제외하기
      
      input_ex = self.df[idx]
      print(input_ex)
      A = len(tokenizer(input_ex)['input_ids'])
      Ch_Attn_weights = Ch_Attn_weights[1:A-1, 1:A-1]
      Num_times = torch.argmax(Ch_Attn_weights, dim =1).tolist()
      cnt = Counter(Num_times)
      Target_Number = cnt.most_common(1)[0][0]
      Final = Real_Token_Dataset(self.df, self.tokenizer)[idx][Target_Number]
      
      return Final
