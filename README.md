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

### instructions for use：ML code
1. **Importing Libraries**: This code starts by importing the necessary Python libraries, including NumPy, Pandas, Matplotlib, Seaborn, and various modules from Scikit-Learn (sklearn). These libraries are essential for data processing, modeling, and evaluation.
2. **Reading Data**: The code reads a dataset from a CSV file named 'date-1.csv' using Pandas' `read_csv` function. It specifies the encoding as 'GBK' to handle potential character encoding issues in the dataset.
3. **Feature and Target Extraction**: It extracts the features (independent variables) and the target variable (dependent variable) from the dataset. Features are stored in the variable `X`, which includes columns 'T', 't', 'R', 'r', 'a', and 'P'. The target variable 'stress' is stored in `y`.
4. **Data Standardization**: The feature data (`X`) is standardized using Scikit-Learn's `StandardScaler`. Standardization ensures that all features have the same scale, which can be important for some machine learning algorithms.
5. **Train-Test Split**: The dataset is split into training and testing sets using Scikit-Learn's `train_test_split` function. 80% of the data is used for training (`X_train` and `y_train`), while 20% is reserved for testing (`X_test` and `y_test`).
6. **Gradient Boosting Regressor**: A Gradient Boosting Regressor model is created using Scikit-Learn's `GradientBoostingRegressor` class. This model will be trained to predict the 'stress' based on the input features.
7. **Hyperparameter Grid**: A grid of hyperparameters is defined in `param_grid`. These hyperparameters include the number of estimators (`n_estimators`), learning rate (`learning_rate`), maximum depth of the trees (`max_depth`), minimum samples required to split a node (`min_samples_split`), and minimum samples required in a leaf node (`min_samples_leaf`).
8. **Grid Search**: Grid search is performed using `GridSearchCV` to find the best combination of hyperparameters that yields the highest R-squared score on the training data.
9. **Best Hyperparameters**: The best hyperparameters from the grid search are printed to the console.
10. **Rebuilding Model**: The model is rebuilt using the best hyperparameters obtained from the grid search.
11. **Model Training**: The model is trained on the training data (`X_train` and `y_train`) using the `fit` method.
12. **Making Predictions**: Predictions are made on the test data (`X_test`) using the trained model, and the predicted values are stored in `y_pred`.
13. **Model Evaluation**: Several evaluation metrics are computed to assess the model's performance, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (`r2`). These metrics help gauge how well the model fits the test data.
14. **Cross-Validation**: Cross-validation is performed using `cross_val_score` to obtain a more robust assessment of the model's performance. The mean R-squared score from cross-validation is printed.
15. **Model Saving**: The trained model is saved as a `.pkl` (Pickle) file named 'model-2.pkl' for future use.
16. **Data Visualization**: A scatter plot is created to visualize the predicted values (`y_pred`) against the true values (`y_test`). A diagonal dashed line represents a perfect prediction.
This code essentially demonstrates the process of training and evaluating a Gradient Boosting Regressor model for predicting 'stress' based on input features. It also showcases the use of hyperparameter tuning and cross-validation to optimize and assess model performance.
![屏幕截图 2023-09-14 203250](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/72adab16-e6d5-4062-8267-262b0549a7fe)

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

### instructions for use：ML code
1. **导入库**：代码首先导入了必要的Python库，包括NumPy、Pandas、Matplotlib、Seaborn，以及Scikit-Learn（sklearn）中的各种模块。这些库用于数据处理、建模和评估。
2. **读取数据**：代码使用Pandas的`read_csv`函数从名为'date-1.csv'的CSV文件中读取数据集。它指定编码为'GBK'，以处理数据集中可能存在的字符编码问题。
3. **提取特征和目标变量**：从数据集中提取特征（自变量）和目标变量（因变量）。特征存储在变量`X`中，包括列'T'、't'、'R'、'r'、'a'和'P'。目标变量'stress'存储在`y`中。
4. **数据标准化**：使用Scikit-Learn的`StandardScaler`对特征数据（`X`）进行标准化。标准化确保所有特征具有相同的尺度，这对于某些机器学习算法很重要。
5. **训练-测试集划分**：使用Scikit-Learn的`train_test_split`函数将数据集划分为训练集和测试集。80%的数据用于训练（`X_train`和`y_train`），而20%保留用于测试（`X_test`和`y_test`）。
6. **梯度提升回归器**：创建了一个梯度提升回归器模型，使用Scikit-Learn的`GradientBoostingRegressor`类。该模型将被训练以基于输入特征预测'stress'。
7. **超参数网格**：在`param_grid`中定义了一组超参数网格，这些超参数包括估计器数量（`n_estimators`）、学习率（`learning_rate`）、树的最大深度（`max_depth`）、拆分节点所需的最小样本数（`min_samples_split`）以及叶节点中所需的最小样本数（`min_samples_leaf`）。
8. **网格搜索**：使用`GridSearchCV`进行网格搜索，以找到在训练数据上获得最高R平方分数的超参数组合。
9. **最佳超参数**：将网格搜索中获得的最佳超参数打印到控制台。
10. **重新构建模型**：使用从网格搜索获得的最佳超参数重新构建模型。
11. **模型训练**：使用`fit`方法在训练数据上训练模型（`X_train`和`y_train`）。
12. **进行预测**：使用训练好的模型在测试数据上进行预测（`X_test`），并将预测值存储在`y_pred`中。
13. **模型评估**：计算多个评估指标，以评估模型的性能，包括均方误差（MSE）、均方根误差（RMSE）和R平方（`r2`）。这些指标有助于衡量模型对测试数据的拟合程度。
14. **交叉验证**：使用`cross_val_score`执行交叉验证，以获得对模型性能的更稳健评估。平均交叉验证R平方分数被打印出来。
15. **保存模型**：将训练好的模型保存为.pkl（Pickle）文件，文件名为'model-2.pkl'，以便将来使用。
16. **数据可视化**：创建散点图以可视化预测值（`y_pred`）与真实值（`y_test`）之间的关系。一条对角虚线代表完美预测。
这段代码实际上展示了如何训练和评估一个梯度提升回归器模型，以基于输入特征预测'stress'。它还展示了超参数调优和交叉验证的使用，以优化和评估模型性能。
![屏幕截图 2023-09-14 203230](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/91b4db19-c0a0-4618-bd88-0473a8d6b04d)
![屏幕截图 2023-09-14 203250](https://github.com/Fan-Tank/Local_Stress_Prediction/assets/76890876/72adab16-e6d5-4062-8267-262b0549a7fe)
