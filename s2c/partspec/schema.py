import json

from s2c.partspec.models import Topology


def topology_json_schema() -> str:
    return json.dumps(Topology.model_json_schema(), indent=2)
