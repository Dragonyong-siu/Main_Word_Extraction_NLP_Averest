#####################################################################################
                ###using BERT's 4th output- attention_weight###
#####################################################################################
def Attention_Weight_BERT(input_ids, token_type_ids, attention_mask):
    config= BertConfig.from_pretrained('/content/gdrive/My Drive/bert_config.json', 
                                       #output_hidden_states=True) 
                                       output_attentions=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config = config)
    out1, _, attn= bert(input_ids, token_type_ids, attention_mask)
    return attn
