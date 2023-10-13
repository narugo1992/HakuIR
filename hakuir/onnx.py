import os
import tempfile

import onnxoptimizer
import onnxsim
import torch

import onnx


def onnx_optimize(model):
    model = onnxoptimizer.optimize(model)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model


def export_model_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True,
                         no_optimize: bool = False):
    example_input = torch.randn(1, 3, 512, 512)

    # if torch.cuda.is_available():
    #     example_input = example_input.cuda()
    #     model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch", 2: 'width', 3: 'height'},
                "output": {0: "batch", 2: 'width', 3: 'height'},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)
