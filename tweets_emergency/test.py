import mlflow
import mlflow.pytorch

breakpoint()
mlflow.pyfunc.load_model('file:///home/ubuntu/mlruns/1/e6c0b16dc2d74c3eb18f25b2514458f7/artifacts/logReg', suppress_warnings=True)