from flask import json

from utils.pathutils import find_path


class Input:
    def __init__(self, id, label=None, service=None, to="output"):
        self.id = id
        self.label = label or id
        self.service = service or ""
        self.to = to


class Output:
    def __init__(self, id, label=None):
        self.id = id
        self.label = label or id


class Arg:
    def __init__(self, name, type="text", label=None, helper="Helper text", value=None):
        self.name = name
        self.type = type
        self.label = label or name
        self.helper = helper
        self.value = value


class Select(Arg):
    def __init__(self, name, options, label=None, helper="Helper text", value=None):
        super().__init__(name, "select", label, helper, value)
        self.options = options


class Component:
    def __init__(self, name, description="", inputs=[Input("input")], outputs=[Output("output")], args=None,
                 configured=True):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.outputs = outputs
        self.args = args or []
        self.configured = configured

    def to_dict(self):
        values = {x.name: x.value for x in self.args if x.value}
        options = dict(group="Custom", icon="RiTyphoonFill", click=None, values=values,
                       args=[x.__dict__ for x in self.args])
        return dict(name=self.name, description=self.description, configured=self.configured,
                    inputs=[x.__dict__ for x in self.inputs],
                    outputs=[x.__dict__ for x in self.outputs], options=options)


def save_extensions(comps):
    output_path = find_path("extensions")
    output = output_path / "components.json"
    with output.open("w") as o:
        json.dump([comp.to_dict() for comp in comps], o, indent=1)
