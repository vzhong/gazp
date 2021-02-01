mkdir -p cache/bert
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -O cache/bert/vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin -O cache/bert/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json -O cache/bert/config.json
