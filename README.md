## Local_Stress_Prediction
### Pressure Vessel Nozzle Local Stress Prediction Software Based on ABAQUS-Machine Learning
The Local_Stress_Prediction package comprises an ABAQUS script and a corresponding plugin (GenerateDataset) for generating datasets, Python code for machine learning and model generation, and a comprehensive stress prediction software (Stress_Prediction-ML). This means that users can refer to these publicly available files to create stress prediction software tailored to specific working conditions and materials.
1. One ABAQUS script and one plugin are publicly available and serve the purpose of generating datasets. This script or plugin is capable of creating nozzle models in batches based on user-provided parameters and subsequently calculating the stress within them. All the calculated data is saved in a .txt file, which can be transformed into datasets in various formats as per requirements. Modifying the script or plugin allows users to obtain datasets for different materials and working conditions, although it is recommended to limit modifications to material parameters and avoid altering other parameters.An example of the dataset is also provided here, which has 2860 samples (date-1.scv).
2. After generating the dataset, machine learning can be performed using Python code to create personalized predictive models. This enables the generation of stress prediction models for any working conditions and materials. These models are designed to carry out predictive tasks, and users have the option to develop graphical user interfaces (GUIs) to provide fully-fledged software that can be used by non-technical individuals.
3. The complete stress prediction software is an illustrative example that can be directly utilized. It employs Q345 as the material and assumes normal temperature as the working condition. On top of this, it uses parameters such as pressure and geometry as features for machine learning to derive the software's predictions.
   
In summary, the ABAQUS plugin or script is used for generating datasets (Data Set, DS), which are then employed for machine learning and model generation. The primary task of this model is stress prediction. For convenience, a graphical user interface (Graphical User Interface, GUI) can be created to develop it into a complete software application.
Firstly, the ABAQUS plugin or script is used to select features for case calculations to generate datasets. Secondly, machine learning Python code is employed to choose an appropriate model for training and evaluate the trained model to ensure accurate stress predictions. Finally, the trained model is embedded in the software, and a GUI is created.
We have already utilized this approach to develop a model for specific working conditions and materials, resulting in a mature stress prediction software (Stress_Prediction-ML). This software is suitable for predicting local stresses in Q345 material pressure vessels under normal temperature conditions.
![1](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/35bd610c-edd0-4543-9827-aedf65b0bcd9)


### instructions for use：GenerateDataset
1. Copy the "GenerateDataset" folder to 'C:\Users\Username\abaqus_plugins'.
2. Open ABAQUS, and you will find "GenerateDataset" under the "Plug-ins" menu.
3. Open "GenerateDataset" and enter the parameters. Click "OK" to complete the batch analysis.
4. In the folder "C:\temp\Local_pipe_analysis", you will find the "MaxMises.txt" file.
5. Copy or convert the data into the desired format for your dataset.
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/4b4ee3bb-7a3a-40c0-81dc-edb464f7a6b8)

### instructions for use：Stress_Prediction-ML
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

## Local_Stress_Prediction
### Pressure Vessel Nozzle Local Stress Prediction Software Based on ABAQUS-Machine Learning
Local_Stress_Prediction包括一个用于生成数据集的ABAQUS脚本和一个同样作用的插件（GenerateDataset）、一段用于机器学习并生成模型的Python代码（Machine learning code）和一个完整的应力预测软件（Stress_Prediction-ML）。这意味着，用户可以参考使用这些已经公开的文件制作专属工况和材料下的应力预测软件。
1.	一个ABAQUS脚本和一个插件是公开的，其功能是用来生成数据集。该脚本或插件会根据用户提供的参数批量的建立接管模型，并计算得出应力。所有的计算数据会生成在一个txt文件中，根据需求可以转变为其他格式的数据集。修改脚本或插件可以获取其他材料和工况下的数据集，对于其修改只限于材料参数，不建议修改其他参数。这里也提供了数据集的示例，它具有2860个样本(date-1.scv)。
2.	生成数据集后可，根据机器学习的Python代码进行机器学习，得到个性化的预测模型。这就可以生成任何工况和材料下的应力预测模型。这个模型将会完成预测任务，用户还可以为其制作GUI并得到完整的软件提供给非技术人员使用。
3.	完整的应力预测软件是一个示例，它可以直接使用，其将Q345作为材料，将常温作为工况。在这基础上将压力、几何等参数作为特征进行机器学习并得出该软件。
   
总的来说，ABAQUS插件或脚本用于生成数据集（Data Set，DS），这个数据集将被用作机器学习并生成模型，这个模型将会完成应力预测的主要任务。为了方便，可以创建图形用户界面（Graphical User Interface，GUI），并制作为完整的软件形式。
首先，利用ABAQUS插件或脚本，选择特征对案例进行计算，以生成数据集。其次，利用机器学习的Python代码，选择适当的模型进行训练，并对训练完成的模型进行评估，以确保其能够准确预测应力。最后，将训练好的模型内嵌在软件中，并创建GUI。
我们已经基于这个方式完成了一种特点工况和材料下的模型，并将其制作成了一个成熟的应力预测软件（Stress_Prediction-ML），其适用于常温下的Q345材料压力容器的局部应力预测。 
![1](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/af7df3c2-d6f3-4c83-bcfa-aa0481a0832b)

### instructions for use：GenerateDataset
1. 将GenerateDataset文件夹复制在“C:\Users\用户名\abaqus_plugins”下。
2. 打开ABAQUS,在Plug-ins下可以找到GenerateDataset。
3. 打开GenerateDataset输入参数，点击OK即可完成批量分析。
4. 在文件夹C:\\temp\\Local_pipe_analysis下找到MaxMises.txt。
5. 将数据复制或转换为您需要的数据集格式即可。
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/00921a70-cf9f-492a-b3ba-baa27769d7b5)

### instructions for use：Stress_Prediction-ML
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
