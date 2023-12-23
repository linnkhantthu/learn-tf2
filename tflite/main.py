from sys import argv
import tflite_runtime.interpreter as tflite


def main(path, *args):
    print(path)
    interpreter = tflite.Interpreter(model_path=path)
    print(interpreter._get_full_signature_list())


print(main(*argv[1:]))
