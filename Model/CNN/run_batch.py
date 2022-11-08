import os

#command = 'python myOtherScript.py ' + sys.argv[1] + ' ' + sys.argv[2]
# os.system(command)

# Configure and run the model

print("\n RUNNING TEST 888 \n")
print("\n", " python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8 ".center(80, '#'), "\n")
os.system("python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8")

print("\n", " python  transformation.py --f finn_QWNet2d_noFC.onnx ".center(80, '#'), "\n")
os.system("python transformation.py --f finn_QWNet2d_noFC.onnx")

print("\n", " python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
os.system(
    "python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir")

print("\n", " python data_postprocessing.py --name test ".center(80, '#'), "\n")
os.system("python data_postprocessing.py --name test_2d")

# END OF SENARIO
'''
print("\n RUNNING TEST 666 \n")
print("\n", " python model.py --t 0 --qdata 6 --qweight 6 --qrelu 6 ".center(80, '#'), "\n")
os.system("python model.py --t 0 --qdata 6 --qweight 6 --qrelu 6")

print("\n", " python transformation.py --f finn_QWNet111.onnx ".center(80, '#'), "\n")
os.system("python transformation.py --f finn_QWNet111.onnx")

print("\n", " python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
os.system(
    "python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir")

print("\n", " python data_postprocessing.py --name test ".center(80, '#'), "\n")
os.system("python data_postprocessing.py --name test_666")

# END OF SENARIO
print("\n RUNNING TEST 888 \n")
print("\n", " python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8 ".center(80, '#'), "\n")
os.system("python model.py --t 0 --qdata 8 --qweight 8 --qrelu 8")

print("\n", " python transformation.py --f finn_QWNet111.onnx ".center(80, '#'), "\n")
os.system("python transformation.py --f finn_QWNet111.onnx")

print("\n", " python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
os.system(
    "python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir")

print("\n", " python data_postprocessing.py --name test_888 ".center(80, '#'), "\n")
os.system("python data_postprocessing.py --name test_888")

# END OF SENARIO

print("\n RUNNING TEST 161616 \n")
print("\n", " python model.py --t 0 --qdata 16 --qweight 16 --qrelu 16 ".center(80, '#'), "\n")
os.system("python model.py --t 0 --qdata 16 --qweight 16 --qrelu 16")

print("\n", " python transformation.py --f finn_QWNet111.onnx ".center(80, '#'), "\n")
os.system("python transformation.py --f finn_QWNet111.onnx")

print("\n", " python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir ".center(80, '#'), "\n")
os.system(
    "python ../../../../finn/src/finn/builder/build_dataflow.py dataflow_build_dir")

print("\n", " python data_postprocessing.py --name test_161616 ".center(80, '#'), "\n")
os.system("python data_postprocessing.py --name test_161616")
'''
