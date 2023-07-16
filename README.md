# Federated Learning 
## Mango_2

### 데이터 살펴보기
- inventory.ipynb (reach_dataset)
- adsfasf.ipynb (잡다한거)

### LSTM LF(~ing)
- LSTM_client.py
- LSTM_server.py
- mat1 and mat2 shape이 안맞는 문제 발생.

### Transformer FL(~ing)
- ref: https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
- ref: https://github.com/nklingen/Transformer-Time-Series-Forecasting
- main.py
- model.py
- Dataloader.py
- Preprocessing.py
- train_teacher_forcing.py
- train_with_sampling.py
- inference.py
- plot.py
- helpers.py
- scaler_item.joblib
- trans_server.py

trans_server가 server.
main이 client.


### Transformer FL2(Fail)
- ref: https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py
- Transformer_client.py
: 돌아가기는 함. 하지만, 단변량 => 다변량 실패.

trans_server가 server.
Transformer_client가 client.