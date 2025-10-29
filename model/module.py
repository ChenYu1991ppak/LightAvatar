""" 图像检测和处理模块 """

import numpy as np
import onnxruntime as ort


class AudioEnCoder(object):

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def __call__(self, bmels: np.ndarray, bf0: np.ndarray) -> np.ndarray:
        bmels_transposed = np.transpose(bmels, (0, 3, 1, 2)).astype(np.float32)
        bf0 = bf0.astype(np.float32)
        inputs = {self.input_names[0]: bmels_transposed, self.input_names[1]: bf0}
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]

    def update_model(self, new_onnx_model_path):
        self.session = ort.InferenceSession(new_onnx_model_path)
