# Local_Stress_Prediction
**A Local Stress Prediction Software for Pressure Vessel Based On ABAQUS-Machine Learning**  
Local_Stress_Prediction consists of an ABAQUS script and a corresponding plugin used for generating a dataset, a Python code segment for machine learning and model generation, and a mature stress prediction software.In summary, the ABAQUS plugin or script is utilized to generate a dataset, which is then employed for machine learning and model creation. This model subsequently performs the prediction task. Initially, prior to machine learning, the ABAQUS plugin or script (GenerateDataset) is used to select features and compute cases to generate the dataset (Data Set, DS). Next, the Python code for machine learning (ML.py) is used to choose an appropriate model for training, evaluate the trained model to ensure its accurate stress prediction capabilities. Finally, the trained model is embedded within the software, and a graphical user interface (GUI) is created. We have applied this approach to develop a model for specific conditions and materials, resulting in a mature stress prediction software (Stress_Prediction-ML). This software is designed for predicting local stresses in Q345 material pressure vessels under ambient temperature conditions.
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/142d0db6-8d8b-4ecc-80a5-f68c86019c26)

1. Both an ABAQUS script and a corresponding plugin, used for generating a dataset, are publicly available, serving the purpose of dataset creation. This script or plugin constructs a nozzle model and conducts calculations to derive stress. Modifications to the script or plugin are limited to material parameters, and changes to other parameters are not recommended. The dataset is generated and stored in a .txt file, which can be transformed into a usable dataset as needed.
2. After generating the dataset, machine learning is performed using Python code (ML.py) to create a customized predictive model. This allows for the generation of stress prediction models for any working condition and material. This model carries out the prediction task, and users can develop a graphical user interface (GUI) to make it a complete software accessible to non-technical individuals.
3. The mature stress prediction software is an example that can be used directly. It utilizes Q345 as the material and operates under ambient temperature conditions. Geometric parameters and pressure are employed as features for machine learning, resulting in the software's functionality.

**instructions for use：GenerateDataset**
1. Copy the "GenerateDataset" folder to 'C:\Users\Username\abaqus_plugins'.
2. Open ABAQUS, and you will find "GenerateDataset" under the "Plug-ins" menu.
3. Open "GenerateDataset" and enter the parameters. Click "OK" to complete the batch analysis.
4. In the folder "C:\temp\Local_pipe_analysis", you will find the "MaxMises.txt" file.
5. Copy or convert the data into the desired format for your dataset.
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/4b4ee3bb-7a3a-40c0-81dc-edb464f7a6b8)

**instructions for use：Stress_Prediction-ML**
1. Press Win+R to open the Run dialog box, then type "cmd" and press Enter. This will open the Command Prompt window.
2. In the Command Prompt window, enter the following commands (install one after another once the previous one is installed):
pip install pyqt5
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
3. Double-click on "Stress_Prediction-ML.exe" to run the program. Enter the corresponding geometric parameters and click OK to obtain the predicted values.
4. You can copy and save the results for each set of input parameters for further analysis.
![GUI-1](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/272e7740-fe9c-46c1-bdc3-c2b0d0729617)

**Local_Stress_Prediction**  
**A Local Stress Prediction Software for Pressure Vessel Based On ABAQUS-Machine Learning**
Local_Stress_Prediction包括一个用于生成数据集的ABAQUS脚本和一个同样作用的插件、一段用于机器学习并生成模型的Python代码和一个成熟的应力预测软件。  
总的来说，ABAQUS插件或脚本是用于生成数据集的，这个数据集将被用作机器学习并生成模型，这个模型将会完成预测的任务。首先，进行机器学习之前，需要利用ABAQUS插件或脚本（GenerateDataset），选择特征对案例进行计算，以生成数据集（Data Set，DS）。其次，利用机器学习的Python代码（ML.py），选择适当的模型进行训练，并对训练完成的模型进行评估，以确保其能够准确预测应力。最后，将训练好的模型内嵌在软件中，并创建图形用户界面（Graphical User Interface，GUI）。我们已经基于这个方式完成了一种特点工况和材料下的模型，并将其制作成了一个成熟的应力预测软件（Stress_Prediction-ML），其适用于常温下Q345材料压力容器的局部应力预测。  
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/8e8a164d-2b5b-42c2-9d6f-24b70caed336)
  
1. 一个用于生成数据集的ABAQUS脚本和一个同样作用的插件都是公开的，其功能是用来生成数据集。该脚本或插件会建立接管模型并进行计算，从而得出应力。修改脚本或插件可以获取特点材料和工况下的数据集。对于其修改只限于材料参数，不建议修改其他参数。数据集会生成在.txt文件中，根据需求可以转变为可用的数据集。
2. 生成数据集后可，根据机器学习的Python代码（ML.py）进行机器学习，得到个性化的预测模型。这就可以生成任何工况和材料下的应力预测模型。这个模型将会完成预测任务，用户还可以为其制作GUI并得到完整的软件共非技术人员使用。
3. 成熟的应力预测软件是一个示例，它可以直接使用，其将Q345作为材料，将常温作为工况。在这基础上将几何参数给压力作为特征进行机器学习并得出该软件。
   
**instructions for use：GenerateDataset**
1. 将GenerateDataset文件夹复制在“C:\Users\用户名\abaqus_plugins”下。
2. 打开ABAQUS,在Plug-ins下可以找到GenerateDataset。
3. 打开GenerateDataset输入参数，点击OK即可完成批量分析。
4. 在文件夹C:\\temp\\Local_pipe_analysis下找到MaxMises.txt。
5. 将数据复制或转换为您需要的数据集格式即可。
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/00921a70-cf9f-492a-b3ba-baa27769d7b5)

**instructions for use：Stress_Prediction-ML**
1. win+R输入cmd后按回车，打开命令提示符窗口。
2. 在命令提示符窗口分输入以下语句（一个安装成功后再安装下一个）。
pip install pyinstaller
pip install pyqt5
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
3. 双击Stress_Prediction-ML.exe运行程序，输入对应几何参数点击OK得出预测值。
4. 每次结果对应输入参数可复制保存，便于分析。
![GUI-1](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/8c48ff97-d456-498f-b6eb-df42dcb43284)
