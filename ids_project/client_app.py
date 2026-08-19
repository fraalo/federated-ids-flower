from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from ids_project.task import (
    Net,
    train,
    evaluate,
    get_parameters,
    set_parameters,
    load_data,
    get_frac_attack,
)


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, self.local_epochs, self.lr)
        # Sent to the server so FedCustom can weight this client's update
        # by how balanced its local benign/attack split is.
        frac_attack = get_frac_attack(self.trainloader)
        return get_parameters(self.net), len(self.trainloader.dataset), {"frac_attack": frac_attack}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]

    trainloader, testloader = load_data(partition_id, batch_size)

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FlowerClient(trainloader, testloader, local_epochs, learning_rate).to_client()


app = ClientApp(client_fn)
