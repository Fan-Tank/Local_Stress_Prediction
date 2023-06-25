# Local_Stress_Prediction
A Local Stress Prediction Software for Pressure Vessel Based On ABAQUS-Machine Learning

Local_Stress_Prediction包括了一个成熟的应力预测软件，同时还公开了一个用于生成数据集的脚本和一个用于生成数据集的ABAQUS插件。
1、成熟的应力预测软件可以直接使用，其将Q345作为材料，将常温作为工况。在这基础上将几何参数给压力作为特征进行机器学习并得出该软件。总的来说，其适用于常温下Q345材料压力容器的局部应力预测。
2、一个用于生成数据集的脚本和一个用于生成数据集的ABAQUS插件都是公开的，其功能是用来生成数据集。修改脚本或插件并允许可以获取个性化的数据集，不限材料和工况。
该脚本或插件会建立接管模型并进行计算，从而得出应力，对于其修改只限于材料参数，不建议修改其他参数。数据集会生成在.txt文件中，根据需求可以转变为可用的数据集。
3、生成个性化的数据集后可根据ML.py进行机器学习，得到个性化的预测模型。这一就可以生成任何工况和材料下的应力预测模型。
首先，进行机器学习之前，需要选择特征并使用ABAQUS软件对案例进行计算，以生成数据集（Data Set，DS）。其次，软件的开发主要集中在训练模型阶段，选择适当的模型进行训练，并对训练完成的模型进行评估，以确保其能够准确预测应力。最后，将训练好的模型内嵌在软件中，并创建图形用户界面（Graphical User Interface，GUI）。
这一研究成果通过ABAQUS-机器学习的方式开发了压力容器局部应力的预测软件，避免了复杂的分析过程和主观性，为快速评估和优化压力容器的设计提供了一个可靠便捷的平台，也为其他的应力预测提供了方式借鉴。

The Local_Stress_Prediction package includes a mature stress prediction software, as well as a script and an ABAQUS plugin for generating datasets, all of which are publicly available.
1. The mature stress prediction software can be used directly and is designed for Q345 material under ambient temperature conditions. It utilizes the pressure as a feature for machine learning to predict local stresses in Q345 pressure vessels at ambient temperature. In summary, it is suitable for predicting local stresses in Q345 pressure vessels under ambient temperature conditions.
2. The script and ABAQUS plugin for generating datasets are both publicly available and serve the purpose of generating datasets. By modifying the script or plugin, personalized datasets can be generated, regardless of the material or operating conditions. These tools establish the computational model and calculate the stresses, with the ability to modify only the material parameters and not recommended for modifying other parameters. The generated datasets are stored in .txt files and can be transformed into usable datasets as needed.
3. After generating personalized datasets, the ML.py script can be used for machine learning to obtain personalized predictive models. This enables the generation of stress prediction models for any operating conditions and materials.
The process involves selecting features and performing calculations in ABAQUS software to generate a dataset (Data Set, DS) before conducting machine learning. The software development primarily focuses on the training phase, where an appropriate model is selected for training and the trained model is evaluated to ensure accurate stress predictions. Finally, the trained model is embedded in the software, and a Graphical User Interface (GUI) is created.
This research achievement has developed a pressure vessel local stress prediction software using ABAQUS and machine learning, eliminating the need for complex analysis procedures and subjectivity. It provides a reliable and convenient platform for the rapid evaluation and optimization of pressure vessel designs, while also serving as a reference for other stress prediction applications.
