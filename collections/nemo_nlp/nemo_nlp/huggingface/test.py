from albert import BertForPreTraining
from configuration_albert import AlBertConfig

model = BertForPreTraining({
    'vocab_size_or_config_json_file':50,
    'hidden_size':50,
    'num_hidden_layers':50,
    'num_attention_heads':12,
    'intermediate_size':120,
    'feedforward_size':525,
    'hidden_act':"gelu",
    'hidden_dropout_prob':0.1,
    'attention_probs_dropout_prob':0.1,
    'max_position_embeddings':10,
    'type_vocab_size':2,
    'initializer_range':0.02,
    'layer_norm_eps':1e-12,
})

print(model)
print(model(torch.LongTensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])))
