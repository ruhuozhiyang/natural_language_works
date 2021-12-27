import torch
import torch.nn as nn
import torch.nn.functional as f


class CnnNet(nn.Module):
  def __init__(self, data_loader, params):
    super(CnnNet, self).__init__()
    embedding_vectors = data_loader.get_loaded_embedding_vectors()

    self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
    # 0<= 相对位置 <=params.pos_dis_limit * 2 + 2
    self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
    self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
    self.dropout = nn.Dropout(params.dropout_ratio)

    feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
    self.conv_s = nn.ModuleList([
      nn.Conv2d(
        in_channels=1,
        out_channels=params.filter_num,
        kernel_size=(k, feature_dim),
        padding=0
      ) for k in params.filters
    ])

    filter_dim = params.filter_num * len(params.filters)
    labels_num = len(data_loader.label2idx)

    self.linear = nn.Linear(filter_dim, labels_num)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, in_data):
    word_emb = self.word_embedding(in_data['sents'])
    pos1_emb = self.pos1_embedding(in_data['pos1s'])
    pos2_emb = self.pos2_embedding(in_data['pos2s'])

    input_feature = torch.cat([word_emb, pos1_emb, pos2_emb], dim=2)
    in_data = input_feature.unsqueeze(1)
    in_data = self.dropout(in_data)
    in_data = [torch.tanh(conv(in_data)).squeeze(3) for conv in self.conv_s]
    in_data = [f.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in in_data]
    sentence_features = torch.cat(in_data, dim=1)
    in_data = self.dropout(sentence_features)
    in_data = self.linear(in_data)
    return in_data
