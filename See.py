# * Parameters dic classification report

  Macro_avg_label = 'macro avg'
  Weighted_avg_label = 'weighted avg'

  Classification_report_labels = []
  Classification_report_metrics_labels = ('precision', 'recall', 'f1-score', 'support')

  for Label in Class_labels:
    Classification_report_labels.append(Label)
  
  Classification_report_labels.append(Macro_avg_label)
  Classification_report_labels.append(Weighted_avg_label)