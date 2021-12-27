import torch
from sklearn.metrics import precision_recall_fscore_support


def evaluate(cnn_net, data_iterator, steps, m_labels):
  cnn_net.eval()
  output_labels = list()
  target_labels = list()

  for _ in range(steps):
    batch_data, batch_labels = next(data_iterator)
    batch_output = cnn_net(batch_data)
    batch_output_labels = torch.max(batch_output, dim=1)[1]
    output_labels.extend(batch_output_labels)
    target_labels.extend(batch_labels)

  p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, zero_division=0,
                                             labels=m_labels, average='micro')
  p_r_f1 = {
    'precision': p_r_f1_s[0] * 100,
    'recall': p_r_f1_s[1] * 100,
    'f1': p_r_f1_s[2] * 100
  }
  return p_r_f1
