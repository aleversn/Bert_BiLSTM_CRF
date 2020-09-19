# %%
from concurrent import futures
import os
import time
import grpc
import json
import uuid
import shutil
import grpc.basic_pb2 as basic_pb2
import grpc.basic_pb2_grpc as basic_pb2_grpc
from ner.trainer import NERTrainer

# %%
class Task(basic_pb2_grpc.TaskServicer):

    def createTrain(self, request, context):
        guid = str(uuid.uuid1()).split('-')[0]
        save_dir = './tmp/{}'.format(guid)
        os.makedirs(save_dir)
        required_list = {
            'bert_config_file': False,
            'pretrained_file': False,
            'train_file': False,
            'vocab_file': False,
            'tags_file': False,
            'eval_file': False
        }
        for r in request:
            cur_name = r.name
            required_list[r.name] = True
            with open(os.path.join(save_dir, cur_name), 'ab') as f:
                f.write(r.file)
        for key in required_list.keys():
            if not required_list[key]:
                shutil.rmtree(save_dir)
                return basic_pb2.Response(status='failed', result='missing {}'.format(required_list[key]))
        return basic_pb2.Response(status='success', result=guid)
    
    def getModel(self, request, context):
        guid = request.guid
        model_id = request.model_id
        save_dir = './tmp/{}'.format(guid)
        model_dir = './save_model'
        if not os.path.isfile(os.path.join(model_dir, 'bert', '{}_bert.pth'.format(model_id))):
            yield basic_pb2.FileBody(file=None, name='missing model file')
        common_files = os.listdir(save_dir)
        for fi in common_files:
            with open(os.path.join(save_dir, fi), mode='rb') as f:
                lines = f.readlines()
            for line in lines:
                yield basic_pb2.FileBody(file=line, name=fi)
        with open(os.path.join(model_dir, 'bert', '{}_bert.pth'.format(model_id)), mode='rb') as f:
                lines = f.readlines()
        for line in lines:
            yield basic_pb2.FileBody(file=line, name=fi)
        with open(os.path.join(model_dir, 'lstm_crf', '{}_lstm_crf.pth'.format(model_id)), mode='rb') as f:
                lines = f.readlines()
        for line in lines:
            yield basic_pb2.FileBody(file=line, name=fi)
    
    def createPredict(self, request, context):
        guid = request.guid
        return basic_pb2.Response(status='success', result=guid)


class Train(basic_pb2_grpc.TrainServicer):

    def train(self, request, context):
        save_dir = './tmp/{}'.format(request.guid)
        trainer = NERTrainer(request.num_epochs, request.num_gpus,
            bert_config_file_name= os.path.join(save_dir, 'bert_config_file'),
            pretrained_file_name= os.path.join(save_dir, 'pretrained_file'),
            hidden_dim= request.hidden_dim,
            train_file_name= os.path.join(save_dir, 'train_file'),
            vocab_file_name= os.path.join(save_dir, 'vocab_file'),
            tags_file_name= os.path.join(save_dir, 'tags_file'),
            eval_file_name= os.path.join(save_dir, 'eval_file'),
            batch_size=request.batch_size,
            eval_batch_size=request.eval_batch_size)
        for info in trainer():
            yield basic_pb2.Response(status='training', result=json.dumps(info))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    basic_pb2_grpc.add_TaskServicer_to_server(Task(), server)
    basic_pb2_grpc.add_TrainServicer_to_server(Train(), server)
    server.add_insecure_port('[::]:8088')
    server.start()

    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()

# %%