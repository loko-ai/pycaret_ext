from loko_extensions.model.components import Component, Input, Output, save_extensions, Select, Arg, Dynamic



def create_components_json():
    predictor_inp = [Input("fit", service="fit", to="fit_output")]
    predictor_out = [Output("fit_output")]
    predictor_args = [Select("task", options=['classification', "regression"], value="classification")]

    dataset_inp = [Input("info", service="datasets", to="info")]
    dataset_out = [Output("info")]
    datasets_comp = Component("datasets", inputs=dataset_inp, outputs=dataset_out, )
    predictor_comp = Component("predictor", inputs=predictor_inp, outputs=predictor_out,args=predictor_args, configured=False)
    save_extensions([datasets_comp, predictor_comp])

