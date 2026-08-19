from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from ids_project.task import Net, get_parameters
from ids_project.FedCustom import FedCustom


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    net = Net()
    initial_parameters = ndarrays_to_parameters(get_parameters(net))

    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    strategy = FedCustom(
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=1,
        min_available_clients=1,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )

    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
