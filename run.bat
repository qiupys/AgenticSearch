@echo off
setlocal enabledelayedexpansion
chcp 936 >nul
title AgenticSearch 一键启动
cd /d "%~dp0"

echo ==============================================
echo   AgenticSearch 一键启动
echo   用法:
echo     run.bat           依赖校验(缓存) + 数据自检 + 启动服务
echo     run.bat build     强制重建全部数据(词库/语料/索引/模型)
echo     run.bat deps      清除依赖缓存，下次启动强制重新校验
echo     run.bat test      运行单元测试
echo ==============================================
echo.

REM ---------- 1. Python 检查 ----------
set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 python。请先安装 Python 3.11+ 并加入 PATH
    pause
    exit /b 1
)

REM ---------- 2. 虚拟环境（存在 .venv 则优先使用） ----------
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if /i "%~1"=="test"  goto :cmd_test
if /i "%~1"=="build" goto :cmd_build
if /i "%~1"=="deps"  goto :cmd_deps

REM ---------- 3. 依赖校验（缓存：requirements.txt 未变则跳过，省 3-6 秒） ----------
set "NEED_DEPS=1"
if exist "data\.deps_ok" (
    set "LAST="
    for /f "delims=" %%i in ('dir /b /o:d "requirements.txt" "data\.deps_ok" 2^>nul') do set "LAST=%%i"
    if /i "!LAST!"==".deps_ok" set "NEED_DEPS=0"
)
if "%NEED_DEPS%"=="1" (
    echo [校验] 检查依赖完整性...
    %PY% -c "import fastapi, uvicorn, yaml, pydantic, sklearn, joblib, numpy, requests, jieba, zhconv" >nul 2>nul
    if errorlevel 1 (
        echo [安装] 检测到缺少依赖，正在安装 requirements.txt ...
        %PY% -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [错误] 依赖安装失败，请检查上方日志
            pause
            exit /b 1
        )
    )
    if not exist "data" mkdir "data"
    echo ok> "data\.deps_ok"
)

REM ---------- 4. LLM 配置（缺失时从模板生成） ----------
if not exist "configs\llm.yaml" (
    copy /y "configs\llm.example.yaml" "configs\llm.yaml" >nul
    echo [提示] 已从模板创建 configs\llm.yaml，请填入真实 api_key，否则 LLM 能力自动降级为规则
)

REM ---------- 5. LLM 本地代理（llm.yaml 指向本地代理时自动拉起） ----------
findstr /c:"127.0.0.1:8180" "configs\llm.yaml" | findstr /r /v "^  *#" >nul 2>nul
if not errorlevel 1 (
    netstat -ano | findstr ":8180" | findstr "LISTENING" >nul 2>nul
    if errorlevel 1 (
        echo [提示] 检测到本地 LLM 代理配置（opencode Zen 免费模型）...
        start "llm-proxy" /min python "%~dp0tools\llm_proxy.py"
        timeout /t 2 /nobreak >nul
    )
)

REM ---------- 6. 首次启动自动构建数据（仅缺数据时执行） ----------
if not exist "data\content\corpus.jsonl" goto :cmd_build

:cmd_start
echo.
echo [启动] AgenticSearch 服务
echo        前端页面: http://localhost:8100/
echo        API 文档: http://localhost:8100/docs
echo        按 Ctrl+C 停止服务
echo.
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8100/"
%PY% -m search.service.app
pause
exit /b 0

:cmd_build
echo [构建] 开始数据构建（约 10-30 分钟，幂等可中断重跑）...
%PY% -m tools.build_stock_lexicon      || goto :fail
%PY% -m tools.build_topic_lexicon      || goto :fail
%PY% -m tools.build_ner_lexicon        || goto :fail
%PY% -m tools.build_content_library    || goto :fail
echo [完成] 数据构建完毕
if /i "%~1"=="build" (
    pause
    exit /b 0
)
goto :cmd_start

:cmd_deps
if exist "data\.deps_ok" del "data\.deps_ok"
echo [完成] 依赖缓存已清除，下次启动将重新校验依赖
pause
exit /b 0

:cmd_test
%PY% -m pytest tests/ -q
set "RC=%errorlevel%"
pause
exit /b %RC%

:fail
echo [错误] 数据构建失败，请查看上方日志
pause
exit /b 1
