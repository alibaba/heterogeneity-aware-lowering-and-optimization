include "model_base.td"
def docs : Docs<"resnet50v2",
                "ONNX",
                "A CNN model(52 layers) uses shortcut connections to achieve higher accuracy when classifying images">;

def Resnet50V2 : Model<1, 1> {
    let docs_ = docs;
    let configs = [Config<1>];
}