
with open("cityscapes_labels.txt", 'r') as w:
  with open("cityscapes_labels_full.txt", 'w') as f:
    all_lines = w.readlines()
    for line in all_lines:
      line = "../data/leftImg8bit_trainvaltest/"+line
      f.write(line)
