@ECHO OFF

SET PYTHON=C:\Python37\python.exe
SET DEEPMEDIC_DIR=E:\Projects\GitHub\deepmedic

SET ORGAN=Kidneys
SET NUM_PATIENTS=20

REM
REM Check CUDA environment
REM
WHERE /Q nvcc.exe
IF %ERRORLEVEL% EQU 0 GOTO has_cudaenv
@ECHO setting CUDA environment
SET PATH=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;%PATH%
:has_cudaenv

SET TASK=%1
SHIFT
SET CONF=%1
SHIFT
GOTO %TASK%
@ECHO ERROR: unknown task %TASK%
GOTO:EOF

:inference
REM "%PYTHON%" ctorg_dm.py -i %ORGAN% -p 0
REM "%PYTHON%" ctorg_dm.py -i %ORGAN% -p 2
"%PYTHON%" ctorg_dm.py -i %ORGAN% -p 3
GOTO:EOF
"%PYTHON%" ctorg_dm.py -i %ORGAN%
DEL /F /Q models.lst
DIR /B /O-D output\saved_models\trainSession%CONF%\*.final*.ckpt.index > models.lst
SET/pz=<models.lst
SET MODEL=%Z:.index=%
@ECHO latest %CONF% model: %MODEL%
"%PYTHON%" "%DEEPMEDIC_DIR%\deepMedicRun" -model ./config/%CONF%/modelConfig.cfg -test ./config/%CONF%/testConfig.cfg -load "output\saved_models\trainSession%CONF%\%MODEL%"
GOTO:EOF

:plot
"%PYTHON%" "%DEEPMEDIC_DIR%\plotTrainingProgress.py" output/logs/trainSession%CONF%.txt -d
GOTO:EOF

:train
"%PYTHON%" ctorg_dm.py -t %ORGAN% -p 0 -n %NUM_PATIENTS%
REM "%PYTHON%" ctorg_dm.py -t %ORGAN% -p 3 -n %NUM_PATIENTS%
REM "%PYTHON%" ctorg_dm.py -t %ORGAN% -n %NUM_PATIENTS%
REM "%PYTHON%" "%DEEPMEDIC_DIR%\deepMedicRun" -model ./config/%CONF%/modelConfig.cfg -train ./config/%CONF%/trainConfig.cfg
GOTO:EOF
