@echo off
echo Step 1: First XeLaTeX compilation...
xelatex -shell-escape -synctex=1 -interaction=nonstopmode main.tex

echo Step 2: BibTeX processing...
if exist main.aux bibtex main

echo Step 3: Second XeLaTeX compilation...
xelatex -shell-escape -synctex=1 -interaction=nonstopmode main.tex

echo Step 4: Final XeLaTeX compilation...
xelatex -shell-escape -synctex=1 -interaction=nonstopmode main.tex

echo Compilation completed!
pause