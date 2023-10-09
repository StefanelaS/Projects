# Project 1: Web scraping and basic summarization


## Overview

In this project articles from RTVSlo.si related to "rusija" or "ukrajina" keyword were extracted and exported into a utf-8 encoded JSON files. Main project 1 code is in the Web_Scraping.ipynb notebook. To run this code please set up the environment as instructed in the "Environment setup" below. Exported JSON files could be found in project-1 GitHub repository by the names data_rusija.json and data_ukrajina.json, for the keywords "rusija" and "ukrajina", respectively. JSON files contain following information for parsed articles: name of the author, date and time of publishment, title, subtitle, headline, text content, information if the article contains a video or not, tags and number of comments on the article. Exported visualization images from Web_Scraping.ipynb could also be found in this repository by the names: visualization_1.png, visualization_2.png, visualization_3.png, visualization_4.png and visualization_5.png.

## Environment setup

- Libraries needed to run Web_Scraping.ipynb notebook are: numpy, matplotlib and selenium.

- Commands for the environment setup:

    conda create -n project1_env python=3.9
    conda activate project1_env
    conda install -y numpy
    conda install -y matplotlib
    conda install -y selenium

 - Or use project1_env.yml file from the same GitHub repository to setup the enviroment with command:
    conda create -f project1_env.yml
    
 - For this project chrome driver was used, which was downloaded from the https://chromedriver.chromium.org/downloads.
    