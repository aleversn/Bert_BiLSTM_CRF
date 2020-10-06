# %%
import grpc
import grpc.basic_pb2 as basic_pb2
import grpc.basic_pb2_grpc as basic_pb2_grpc
from tqdm import tqdm

# %%
def run():
    channel = grpc.insecure_channel('localhost:8088')

    stub = basic_pb2_grpc.TaskStub(channel)
    required_list = [
        'bert_config_file',
        'pretrained_file',
        'train_file',
        'vocab_file',
        'tags_file',
        'eval_file'
    ]

    def read_file():
        with open('./model/chinese_wwm_ext/bert_config.json', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[0])
        with open('./model/chinese_wwm_ext/pytorch_model.bin', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[1])
        with open('./data/news/train.txt', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[2])
        with open('./model/chinese_wwm_ext/vocab.txt', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[3])
        with open('./data/news_tags_list.txt', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[4])
        with open('./data/news/test.txt', mode='rb') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            yield basic_pb2.FileBody(file=line, name=required_list[5])

    response = stub.createTrain(read_file())
    print(response)


# %%
run()

# %%
def train(guid,
          num_epochs,
          num_gpus,
          hidden_dim,
          word_tag_split,
          pattern,
          padding_length,
          batch_size,
          eval_batch_size):
    channel = grpc.insecure_channel('localhost:8088')

    stub = basic_pb2_grpc.TrainStub(channel)
    response = stub.train(basic_pb2.TrainInfo(
        guid=guid, num_epochs=num_epochs, num_gpus=num_gpus,hidden_dim=hidden_dim, word_tag_split=word_tag_split, pattern=pattern, padding_length=padding_length, batch_size=batch_size, eval_batch_size=eval_batch_size))
    for r in response:
        print(r)


# %%
train(guid='bf1fc0f6', num_epochs=10, num_gpus=[0, 1, 2, 3], hidden_dim=150, word_tag_split=' ', pattern='， O', padding_length=50, batch_size=250, eval_batch_size=64)

# %%
def get_model():
    channel = grpc.insecure_channel('localhost:8088')

    stub = basic_pb2_grpc.TaskStub(channel)

    response = stub.getModel(basic_pb2.GetInfo(guid='bf1fc0f6', model_id=''))