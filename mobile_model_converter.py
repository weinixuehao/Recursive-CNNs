import model
import shutil
from pathlib import Path
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile


def load_doc_model(checkpoint_dir, dataset):
    _model = model.ModelFactory.get_model("resnet", dataset)
    _model.load_state_dict(torch.load(checkpoint_dir, map_location="cpu"))
    return _model


if __name__ == "__main__":
    models = [
        {
            "name": "corner_model",
            "model": load_doc_model(
                "/Users/imac-1/Downloads/output/13122021/CornerModel_0/CornerModelcorner_resnet.pb",
                "corner",
            ),
        },
        {
            "name": "doc_model",
            "model": load_doc_model(
                "/Users/imac-1/Downloads/output/13122021/DocModel_0/DocModeldocument_resnet.pb",
                "document",
            ),
        },
    ]

    output = "output/model/mobile"
    shutil.rmtree(output, ignore_errors=True)
    path = Path(output)
    path.mkdir(parents=True)
    for item in models:
        _model = item["model"]
        # _model.eval()
        # scripted = torch.jit.script(_model)

        # optimized_model = optimize_for_mobile(scripted, backend='metal')
        # print(torch.jit.export_opnames(optimized_model))
        # optimized_model._save_for_lite_interpreter(f'{output}/{item["name"]}_metal.ptl')
        
        # scripted_model = torch.jit.script(_model)
        # optimized_model = optimize_for_mobile(scripted_model, backend='metal')
        # print(torch.jit.export_opnames(optimized_model))
        # optimized_model._save_for_lite_interpreter(f'{output}/{item["name"]}_metal.pt')

        torch.save(_model, f'{output}/{item["name"]}.pth')
