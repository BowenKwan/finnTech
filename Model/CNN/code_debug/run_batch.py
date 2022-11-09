import os

# Configure and run the model

print("\n RUNNING TEST 888 \n")
print("\n", " python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8 ".center(80, '#'), "\n")
os.system("python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8")

print("\n", " python  transformation.py --f finn_QWNet2d_noFC.onnx ".center(80, '#'), "\n")
os.system("python transformation.py --f finn_QWNet2d_noFC.onnx")

print("\n", " python ../../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
os.system(
    "python ../../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir")
# END OF SENARIO