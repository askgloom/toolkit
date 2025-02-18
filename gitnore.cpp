# Build directories
build/
out/
cmake-build-*/
.cmake/
CMakeFiles/
CMakeCache.txt
CMakeUserPresets.json

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
.vs/
*.suo
*.user
*.userosscache
*.sln.docstates
*.vcxproj.filters

# Compiled files
*.o
*.obj
*.dll
*.so
*.dylib
*.lib
*.a
*.exe
*.out
*.app
*.pdb
*.ilk
*.exp

# Generated files
version.hpp
compile_commands.json
.clangd/
.cache/

# Dependencies
libs/*/
!libs/CMakeLists.txt
!libs/README.md
deps/
vcpkg/

# Testing
Testing/
CTestTestfile.cmake
DartConfiguration.tcl
test_detail.xml

# Documentation
docs/build/
docs/html/
docs/xml/
docs/latex/
doxygen_warnings.log

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
venv/
.env/
.venv/
pip-log.txt

# Node.js
node_modules/
npm-debug.log
yarn-debug.log
yarn-error.log
package-lock.json

# macOS specific
.DS_Store
.AppleDouble
.LSOverride
._*

# Windows specific
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/

# Linux specific
*~
.fuse_hidden*
.directory
.Trash-*

# Project specific
# Add any project-specific files to ignore here
gloom.log
*.gloom
.gloom/
gloom_cache/

# Coverage reports
*.gcno
*.gcda
*.gcov
coverage/
coverage.info
coverage.xml

# Profiling data
gmon.out
callgrind.out.*
perf.data*

# Backup files
*.bak
*.backup
*~
*.old

# Temporary files
*.tmp
*.temp
.*.swp
.*.swo

# Environment files
.env
.env.local
.env.*.local

# Binary files
bin/
dist/

# Editor specific
.project
.classpath
.settings/
*.sublime-workspace
*.sublime-project

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# OS generated files
.DS_Store?
.Spotlight-V100
.Trashes
Icon?
ehthumbs.db
Thumbs.db

# Custom ignores for Gloom
# Web related
web/dist/
web/node_modules/
web/.cache/

# Model files
*.onnx
*.pt
*.h5
*.weights
*.model

# Data files
*.dat
*.bin
*.vec
*.npy
*.npz
*.pkl
*.pickle

# Configuration files
config.local.json
settings.local.json