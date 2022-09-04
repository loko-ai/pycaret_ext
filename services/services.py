import functools

from flask import Flask, request, jsonify

from business.components import Input, Component, save_extensions, Output, Arg, Select
import pycaret.datasets as datasets

app = Flask("")

datasets_c = Component("datasets", inputs=[Input("input", service="datasets")])
pred = Component("predictor", inputs=[Input("fit", service="fit", to="fit_output")], outputs=[Output("fit_output")],
                 args=[Select("task", options=['classification', "regression"], value="classification")])
print(pred.to_dict())
save_extensions([datasets_c, pred])

import pycaret.classification as pc
import pycaret.regression as rg

all_datasets = datasets.get_data('index', verbose=False)


def dec(f):
    @functools.wraps(f)
    def temp():
        args = request.json.get('args')
        value = request.json.get("value")
        return f(value, args)

    return temp


@app.route("/fit", methods=["POST"])
@dec
def fit(value, args):
    task = args.get("task")

    data = datasets.get_data(value, verbose=False)
    if task == "classification":
        pc.setup(data, target=data.columns[-1], silent=True)
        best = pc.compare_models()
        df = pc.pull()

        return jsonify([str(best)] + df.to_dict("record"))
    if task == "regression":
        rg.setup(data, target=data.columns[-1], silent=True)
        best = rg.compare_models()
        df = rg.pull()

        return jsonify(dict(best=str(best), table=df.to_dict("record")))
    raise Exception(f"Task {task} unknown")


@app.route("/datasets", methods=["POST"])
@dec
def get_datasets(value, args):
    return jsonify(all_datasets[["Dataset", "Default Task"]].to_dict("record"))


if __name__ == "__main__":
    app.run("0.0.0.0", 8080)
