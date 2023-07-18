from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    mses = [num_examples * m["MSE"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"MSE": sum(mses) / sum(examples)}

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)