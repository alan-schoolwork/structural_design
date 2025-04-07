import array
import json
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from jax import Array
from jax._src.array import ArrayImpl
from pintax import unitify
from pintax._core import UnitTracer

from lib.utils import jit
from midterm.build_graph import build_graph
from midterm.main import solve_forces_final

reg = jsonpickle.handlers.register(ArrayImpl)
assert reg is not None


@reg
class arr_handler(jsonpickle.handlers.ArrayHandler):
    def flatten(self, obj, data: dict[str, str]):
        obj = np.array(obj).tolist()
        data["value"] = self.context.flatten(obj, reset=False)
        return data


@unitify
def get_data():
    graph = build_graph()
    ans = jsonpickle.encode(graph, indent=4, make_refs=False, warn=True)
    assert isinstance(ans, str)

    file = Path(__file__).parent / "data" / "structure.json"
    file.write_text(ans)


if __name__ == "__main__":
    get_data()
