import argparse
import shutil
from pathlib import Path
import os
import torch
from tinynn.converter import TFLiteConverter
import model

parser = argparse.ArgumentParser()
parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                    default="../cornerModelWell")
parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                    default="../documentModelWell")

def load_doc_model(checkpoint_dir, dataset):
    _model = model.ModelFactory.get_model("resnet", dataset)
    _model.load_state_dict(torch.load(checkpoint_dir, map_location="cpu"))
    return _model

if __name__ == "__main__":
    args = parser.parse_args()
    models = [
        {
            "name": "corner_model",
            "model": load_doc_model(
                args.cornerModel,
                "corner",
            ),
        },
        {
            "name": "doc_model",
            "model": load_doc_model(
                args.documentModel,
                "document",
            ),
        },
    ]

    out_dir = "output_tflite"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.mkdir(out_dir)
    for item in models:
        _model = item["model"]
        _model.eval()

        dummy_input = torch.rand((1, 3, 32, 32))
        modelPath = f'{out_dir}/{item["name"]}.tflite'
        converter = TFLiteConverter(_model, dummy_input, modelPath)
        converter.convert()
        # scripted = torch.jit.script(_model)

        # optimized_model = optimize_for_mobile(scripted, backend='metal')
        # print(torch.jit.export_opnames(optimized_model))
        # optimized_model._save_for_lite_interpreter(f'{output}/{item["name"]}_metal.ptl')
        
        # scripted_model = torch.jit.script(_model)
        # optimized_model = optimize_for_mobile(scripted_model, backend='metal')
        # print(torch.jit.export_opnames(optimized_model))
        # optimized_model._save_for_lite_interpreter(f'{output}/{item["name"]}_metal.pt')

        # torch.save(_model, f'{output}/{item["name"]}.pth')
