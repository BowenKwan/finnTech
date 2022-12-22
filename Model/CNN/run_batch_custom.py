import os
import shutil

# Configure and run the model
# 5 cases for 2d CNN 
#CNNcase=[1,2,4,5]  # case 3 is skipped as it is too big for FINN to compile
# Bit width for data, weight and relu
#CNNbit=[3,6,8,12]  

CNNcase=[3]
CNNbit=[3]
#CNNbit=[12]

#hardware directory is used to store all FINN IP
if os.path.exists("./hardware/")==False:
    os.mkdir("./hardware/")

# iterate over the 5 case 
for k in range(len(CNNcase)):
    
    if os.path.exists("./hardware/case%d"%(CNNcase[k]))==False:
        os.mkdir("./hardware/case%d"%(CNNcase[k]))
    
    # iterate over the different bit width 
    for j in range(len(CNNbit)):
        
        if os.path.exists("./hardware/case%d/%dbit"%(CNNcase[k],CNNbit[j]))==False:
            os.mkdir("./hardware/case%d/%dbit"%(CNNcase[k],CNNbit[j]))
        print("\n RUNNING TEST %d for case %d\n"%(CNNbit[j],CNNcase[k]))
        
        # Build model and generate onnx file
        print("\n", (" python model.py --t 0 --m %d --qdata %d --qweight %d --qrelu %d "%(CNNcase[k],CNNbit[j],CNNbit[j],CNNbit[j])).center(80, '#'), "\n")
        os.system("python CNNmodel_case.py --t 0 --m %d --qdata %d --qweight %d --qrelu %d "%(CNNcase[k],CNNbit[j],CNNbit[j],CNNbit[j]))

        # Clear up the onnx file 
        print("\n", (" python  transformation.py --f finn_QWNet2d_CNN_case%d.onnx"%(CNNcase[k])).center(80, '#'), "\n")
        os.system("python transformation.py --f finn_QWNet2d_CNN_case%d.onnx"%(CNNcase[k]))


        for i in range(7):
            # set build for different folding (unrolling) config
            
            #shutil.copyfile("./configuration_files/case%d/folding_config_case%d_%dbit_%d.json"%(CNNcase[k],CNNcase[k],CNNbit[j],i),
            #            "./dataflow_build_dir/folding_config.json")

            #shutil.copyfile("./configuration_files/case%d/folding_config_case%d_%dbit_%d.json"%(CNNcase[k],CNNcase[k],CNNbit[j],i),
            #            "./dataflow_build_dir_custom/folding_config.json")
            
            #build hardware 
            print("\n", " cd dataflow_build_dir_custom/ ".center(80, '#'), "\n")
            print("\n", " python -mpdb -cc -cq build.py ".center(80, '#'), "\n")
            os.chdir("./dataflow_build_dir_custom/")
            print("\n", (" NOW RUNNING Config %d TEST %d for case %d\n"%(i,CNNbit[j],CNNcase[k])).center(40, '#'), "\n")
            os.system("python -mpdb -cc -cq build.py")
            os.chdir("..")

            #store latency and resource usage in results.csv
            print("\n", (" python data_postprocessing_custom.py --name CNN_case%d_%dbit_%d"%(CNNcase[k],CNNbit[j],i)).center(80, '#'), "\n")
            os.system("python data_postprocessing_custom.py --name CNN_case%d_%dbit_%d"%(CNNcase[k],CNNbit[j],i))
            
            #save the build 
            print("saving build")
            print("cp -r ./dataflow_build_dir_custom ./hardware/case%d/%dbit/config%d"%(CNNcase[k],CNNbit[j],i))
            os.system("cp -r ./dataflow_build_dir_custom ./hardware/case%d/%dbit/config%d"%(CNNcase[k],CNNbit[j],i))
# END OF SENARIO
