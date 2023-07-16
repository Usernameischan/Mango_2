import flwr as fl


if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )
    
    # def weighted_average(metrics):
    #     # Multiply accuracy of each client by number of examples used
    #     mses = [num_examples * m["MSE"] for num_examples, m in metrics]
    #     examples = [num_examples for num_examples, _ in metrics]
    #     # Aggregate and return custom metric (weighted average)
    #     return {"MSE": sum(mses) / sum(examples)}

    # strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )