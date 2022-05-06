# IPyLabel
## Interactive IPython widget to label saccades in eyetracking data
![alt text](https://github.com/timnaher/saclabel/blob/main/other/screenshot.png)

### Instructions
1) Clone the repo
2) install the package with
` pip install -e . `

3) label saccades


### Usage
1) move your eye data into IPyLabel/data. The files should be names X_train.csv and Y_train.csv with rows as trials and colums as time (default Fs = 1000)
2) Either run the juptyter notebook /jupyer lab or just render the output with voila using `voila <path-to-notebook> --theme=dark` Dark theme works best for contrast
