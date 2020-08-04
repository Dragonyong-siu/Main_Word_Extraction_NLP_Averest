def KFM_BERT_Embedding(input_ids, token_type_ids, attention_mask):
    config= BertConfig.from_pretrained('/content/gdrive/My Drive/bert_config.json') 
                                       #output_hidden_states=True) 
                                       #output_attentions=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config = config)
    out1, _= bert(input_ids, token_type_ids, attention_mask)
    word_embedding = out1
    return word_embedding 
