class KFMDataset(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_len = 96):
    self.df = df
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                   do_lower_case=True)
    self.max_len = max_len

  def __len__(self):

    return len(self.df)

  def __getitem__(self, idx):

    df = str(self.df[idx])
    df = " ".join(df.split())
    
    #먼저 숫자화된 자료얻기
    input_ids = self.tokenizer.encode(df)
    token_type_ids = self.tokenizer(df).token_type_ids
    attention_mask = self.tokenizer(df).attention_mask
    
    padding_length = self.max_len - len(input_ids)
    
    #BERT에 알맞는 길이로 변환하기
    input_ids = input_ids + [1] * padding_length
    token_type_ids = token_type_ids + [0] * padding_length
    attention_mask = attention_mask + [0] * padding_length

    #Tensor로 만들어주기
    input_ids = torch.Tensor(input_ids).long()
    token_type_ids = torch.Tensor(token_type_ids).long()
    attention_mask = torch.Tensor(attention_mask).long()

    return (input_ids, token_type_ids, attention_mask)
