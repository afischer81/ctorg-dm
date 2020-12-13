REM
REM Check CUDA environment
REM
WHERE nvcc.exe
IF ERRORLEVEL 0 GOTO has_cudaenv
SET PATH=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;%PATH%
:has_cudaenv

SET PYTHON=C:\Python37\python.exe
SET DEEPMEDIC_DIR=E:\Projects\GitHub\deepmedic

SET TASK=%1
SHIFT
GOTO %TASK%
@ECHO ERROR: unknown task %TASK%
GOTO:EOF

:inference
SET MODEL=output\saved_models\trainSessionTiny\tinyCnn.trainSessionTiny.final.2020-11-28.12.23.35.841906.model.ckpt
"%PYTHON%" ctorg_dm.py -i Liver
"%PYTHON%" "%DEEPMEDIC_DIR%\deepMedicRun" -model ./config/%1/modelConfig.cfg -test ./config/%1/testConfig.cfg  -load "%MODEL%"
GOTO:EOF

:plot
"%PYTHON%" "%DEEPMEDIC_DIR%\plotTrainingProgress.py" output/logs/trainSessionTiny.txt -d
GOTO:EOF

:train
"%PYTHON%" ctorg_dm.py -t Liver
"%PYTHON%" "%DEEPMEDIC_DIR%\deepMedicRun" -model ./config/%1/modelConfig.cfg -train ./config/%1/trainConfig.cfg
GOTO:EOF
