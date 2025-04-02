#!/bin/bash

# Script to build the paper.pdf file

TEX_FILE="paper"

if [ "$1" == "make" ]; then
    # Compile LaTeX
    echo "change directory to paper"
    cd paper

    pdflatex -shell-escape "${TEX_FILE}.tex"
    
    # Run BibTeX
    bibtex "${TEX_FILE}"
    
    # Compile LaTeX twice more to update references
    pdflatex -shell-escape "${TEX_FILE}.tex"
    pdflatex -shell-escape "${TEX_FILE}.tex"
    
    echo "Compilation complete. Check the output PDF."
elif [ "$1" == "clean" ]; then
    # Remove auxiliary files
    cd paper
    rm -rf *.aux *.bbl *.blg *.log *.out *.pdf *.toc *.lof *.lot svg-inkscape
    
    echo "Cleaned up generated files."\else
    echo "Invalid argument. Use 'make' to compile or 'clean' to remove generated files."
    exit 1
fi