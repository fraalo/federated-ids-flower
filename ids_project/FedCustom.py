from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters


class FedCustom(FedAvg):
    """FedAvg variant that weights each client's update by how balanced its
    local benign/attack split is, on top of the usual sample-count weight.

    Motivation: CIC-IDS2017 is dominated by benign traffic. A client whose
    local partition is almost entirely benign contributes an update biased
    towards ignoring the attack class -- naive FedAvg would still let it
    weigh in proportional to its sample count. Down-weighting those clients
    (via balance_weight, which peaks at 1.0 for a 50/50 split and goes to 0
    for a fully one-sided split) aims to keep the aggregated model attentive
    to the minority (attack) class.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict]:

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        weights_results = []
        for _, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)

            frac_attack = fit_res.metrics.get("frac_attack", 0.5)
            frac_benign = 1.0 - frac_attack
            balance_weight = 2 * min(frac_attack, frac_benign)

            custom_weight = fit_res.num_examples * balance_weight
            weights_results.append((client_params, custom_weight))

        # Layer-wise weighted average across clients.
        total_weight = sum(w for _, w in weights_results)
        aggregated_ndarrays = []
        for i in range(len(weights_results[0][0])):
            layer_sum = 0
            for p, w in weights_results:
                layer_sum += p[i] * w
            aggregated_ndarrays.append(layer_sum / total_weight)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        return parameters_aggregated, {}
