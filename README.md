# Local_Stress_Prediction
A Local Stress Prediction Software for Pressure Vessel Based On ABAQUS-Machine Learning

Local_Stress_Prediction包括了一个成熟的应力预测软件、一个用于生成数据集的ABAQUS脚本和一个同样作用的插件、用于机器学习并生成模型的Python代码。  
总的来说，ABAQUS插件或脚本是用于生成数据集的，这个数据集将被用作机器学习并生成模型，这个模型将会完成预测的任务。我们已经基于这个方式完成了一种特点工况和材料下的模型，并将其制作成了一个成熟的应力预测软件，其适用于常温下Q345材料压力容器的局部应力预测。  
1、成熟的应力预测软件可以直接使用，其将Q345作为材料，将常温作为工况。在这基础上将几何参数给压力作为特征进行机器学习并得出该软件。  
2、一个用于生成数据集的脚本和一个用于生成数据集的ABAQUS插件都是公开的，其功能是用来生成数据集。修改脚本或插件并允许可以获取个性化的数据集，不限材料和工况。  
该脚本或插件会建立接管模型并进行计算，从而得出应力，对于其修改只限于材料参数，不建议修改其他参数。数据集会生成在.txt文件中，根据需求可以转变为可用的数据集。  
3、生成个性化的数据集后可根据ML.py进行机器学习，得到个性化的预测模型。这一就可以生成任何工况和材料下的应力预测模型。  
首先，进行机器学习之前，需要选择特征并使用ABAQUS软件对案例进行计算，以生成数据集（Data Set，DS）。其次，软件的开发主要集中在训练模型阶段，选择适当的模型进行训练，并对训练完成的模型进行评估，以确保其能够准确预测应力。最后，将训练好的模型内嵌在软件中，并创建图形用户界面（Graphical User Interface，GUI）。  
这一研究成果通过ABAQUS-机器学习的方式开发了压力容器局部应力的预测软件，避免了复杂的分析过程和主观性，为快速评估和优化压力容器的设计提供了一个可靠便捷的平台，也为其他的应力预测提供了方式借鉴。  

instructions for use：GenerateDataset  
1.将GenerateDataset文件夹复制在“C:\Users\用户名\abaqus_plugins”下。  
2.打开ABAQUS,在Plug-ins下可以找到GenerateDataset。  
2.打开GenerateDataset输入参数，点击OK即可完成批量分析。  
4.在文件夹C:\\temp\\Local_pipe_analysis下找到MaxMises.txt。  
5.将数据复制或转换为您需要的数据集格式即可。  
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/00921a70-cf9f-492a-b3ba-baa27769d7b5)

  
instructions for use：Stress_Prediction-ML  
1.win+R输入cmd后按回车，打开命令提示符窗口。  
2.在命令提示符窗口分输入以下语句（一个安装成功后再安装下一个）。  
pip install pyinstaller
pip install pyqt5
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn

3.双击Stress_Prediction-ML.exe运行程序，输入对应几何参数点击OK得出预测值。
4.每次结果对应输入参数可复制保存，便于分析。
![GUI-1](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/8c48ff97-d456-498f-b6eb-df42dcb43284)


The Local_Stress_Prediction package includes a mature stress prediction software, as well as a script and an ABAQUS plugin for generating datasets, all of which are publicly available.
1. The mature stress prediction software can be used directly and is designed for Q345 material under ambient temperature conditions. It utilizes the pressure as a feature for machine learning to predict local stresses in Q345 pressure vessels at ambient temperature. In summary, it is suitable for predicting local stresses in Q345 pressure vessels under ambient temperature conditions.
2. The script and ABAQUS plugin for generating datasets are both publicly available and serve the purpose of generating datasets. By modifying the script or plugin, personalized datasets can be generated, regardless of the material or operating conditions. These tools establish the computational model and calculate the stresses, with the ability to modify only the material parameters and not recommended for modifying other parameters. The generated datasets are stored in .txt files and can be transformed into usable datasets as needed.
3. After generating personalized datasets, the ML.py script can be used for machine learning to obtain personalized predictive models. This enables the generation of stress prediction models for any operating conditions and materials.
The process involves selecting features and performing calculations in ABAQUS software to generate a dataset (Data Set, DS) before conducting machine learning. The software development primarily focuses on the training phase, where an appropriate model is selected for training and the trained model is evaluated to ensure accurate stress predictions. Finally, the trained model is embedded in the software, and a Graphical User Interface (GUI) is created.
This research achievement has developed a pressure vessel local stress prediction software using ABAQUS and machine learning, eliminating the need for complex analysis procedures and subjectivity. It provides a reliable and convenient platform for the rapid evaluation and optimization of pressure vessel designs, while also serving as a reference for other stress prediction applications.

instructions for use：GenerateDataset
1. Copy the "GenerateDataset" folder to 'C:\Users\Username\abaqus_plugins'.
2. Open ABAQUS, and you will find "GenerateDataset" under the "Plug-ins" menu.
3. Open "GenerateDataset" and enter the parameters. Click "OK" to complete the batch analysis.
4. In the folder "C:\temp\Local_pipe_analysis", you will find the "MaxMises.txt" file.
5. Copy or convert the data into the desired format for your dataset.
![image](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/4b4ee3bb-7a3a-40c0-81dc-edb464f7a6b8)


instructions for use：Stress_Prediction-ML
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

