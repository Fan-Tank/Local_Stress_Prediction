# Import necessary modules
from abaqusGui import getAFXApp, Activator, AFXMode
from abaqusConstants import ALL
import os

# Get the absolute path and directory of the current script file
thisPath = os.path.abspath(__file__)
thisDir = os.path.dirname(thisPath)

# Get the plugin toolset of the current ABAQUS program
toolset = getAFXApp().getAFXMainWindow().getPluginToolset()

# Register a GUI menu button
toolset.registerGuiMenuButton(
    buttonText='GenerateDataset',  # Text displayed on the button
    object=Activator(os.path.join(thisDir, 'GenerateDatasetDB.py')),  # Activator object for the plugin
    kernelInitString='import GenerateDataset',  # Initialization string for the plugin
    messageId=AFXMode.ID_ACTIVATE,  # Message ID
    icon=None,  # Icon (set to None here)
    applicableModules=ALL,  # Applicable modules (all modules)
    version='N/A',  # Version information
    author='N/A',  # Author information
    description='N/A',  # Description
    helpUrl='N/A'  # Help URL
)
