import functools

from flask import Flask, request, jsonify

import pycaret.datasets as datasets
from loko_extensions.business.decorators import extract_value_args
from extensions.components import create_components_json
from utils.logger_utils import logger

app = Flask("")




import pycaret.classification as pc
import pycaret.regression as rg

create_components_json()
all_datasets = datasets.get_data('index', verbose=False)



@app.route("/fit", methods=["POST"])
@extract_value_args(_request=request)
def fit(value, args):
    task = args.get("task")
    logger.debug(f"task: {task}")
    data = datasets.get_data(value, verbose=False)

    if task == "classification":
        logger.debug("data setup")
        pc.setup(data, target=data.columns[-1], silent=True)
        logger.debug("comparing models")
        best = pc.compare_models()
        df = pc.pull()

        return jsonify([str(best)] + df.to_dict("record"))
    if task == "regression":
        logger.debug("data setup")
        rg.setup(data, target=data.columns[-1], silent=True)
        logger.debug("comparing models")
        best = rg.compare_models()
        df = rg.pull()

        return jsonify(dict(best=str(best), table=df.to_dict("record")))
    raise Exception(f"Task {task} unknown")


@app.route("/datasets", methods=["POST"])
@extract_value_args(_request=request,file=False)
def get_datasets(value, args):
    return jsonify(all_datasets[["Dataset", "Default Task"]].to_dict("record"))


if __name__ == "__main__":
    app.run("0.0.0.0", 8080)