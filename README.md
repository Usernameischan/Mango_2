# Federated Learning 
## Mango_2


### Strategy in flwr
- initialize_params
: Input model's parameters(shape)

- configure fit <br>
: round에서 어떤 client를 선택할 지, 어떤 지시(학습)를 할 것인지를 정한다. <br>
: ClinetManager -> 사용가능한 Client를 무작위로 Sampling.

- aggregate fit <br>
: configure fit에서 선택하여 학습시킨 Client의 결과를 집계한다.

- configure evaluate <br>
: 각 Client를 어떤 식으로 evaluate할 지를 정한다.

- aggregate evaluate <br>
: configure evaluate의 결과를 집계한다.

- evaluate <br>
Server에서의 Model Parameter(Global model Params)를 evaluate한다.


### Vanilla Split Neural Network (SplitNN)

- Step by Step <br>
1. pick a model and a worker with data
2. segment the model <br>
: model의 layer을 분할
3. distribute the segments <br>
: 분할한 segment을 worker에 분할
4. Data is fed by the segment sequentially. <br> 
:Data는 분할된 segment에 의해 기존 model을 나눈 순서대로 feed forward를 수행된다.
5. Back propagation은 feed forward의 반대 방향으로 진행된다.


### Vertical SplitNN

- Vanilla SplitNN과 크게 다르지 않으나, Output을 출력하는 층에서 각 Client가 Worker을 거친 결과(Signal)을 병합하여 feed를 수행하는 점에서 차이가 존재한다.

- 여기서 Worker는 학습을 수행할 수 있는 환경을 말하는 듯. <br>
Vertically Partitioned Data의 소유자를 Client라 하고, <br> 
Output을 담당하는 하단 네트워크를 N개의 Segment로 분할하였다면, <br>
Worker의 수 = Client의 수 + N으로 볼 수 있음. 

- 용님이 참고한 github 코드(에리카님이 올리신 코드)도 이 Vertical SplitNN을 참고함. (해당 코드에서 언급.. 하지만, 어느 부분에서 구현된 것인지는 잘 모르겠음...)