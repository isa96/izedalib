
# IZEDALIB 

Ini library dibuat untuk membantu analisis data (exploratory data analisis).


[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)


## Requirements

- pandas
- scipy
- numpy
- matplotlib
- seaborn

## Features

#### Get all items

| Library | Class       | Function          | Params                                                             | Return                                   | Description                                                                   |
| ------- | ----------- | ----------------- | ------------------------------------------------------------------ | ---------------------------------------- | ----------------------------------------------------------------------------- |
| scan    | Scan        | scan              | root, pattern=`'*.csv'`, verbose=0                                 | full-path-files,  (path, subdirs, files) | get spesific path files in folder and sub-folder                              |
| analisis| Signifikan  | KolmogorovSmirnov | control, test                                                      | value and information                    | Hypothesis test using K-S method                                              |
|         |             | PSI               | control, test                                                      | value and information                    | Hypothesis test using PSI method                                              |
|         |             | cohend            | control, test                                                      | value and information                    | Calculate effect size using Cohen's method                                    |
| plot2D  | Tampilan    | Reset             | -                                                                  | -                                        | Reset rcParams plot (matplotlib)                                              |
|         |             | Kertas            | loc='best', classic=True, figsize=[13.3, 8]                        | figure, axes                             | Update rcParams with custom matplotlib theme plot                             |
| plot3D  | Tampilan    | Reset             | -                                                                  | -                                        | Reset rcParams plot (matplotlib)                                              |
|         |             | Kertas            | loc='best', classic=True, figsize=[13.3, 8]                        | figure, axes                             | Update rcParams with custom matplotlib theme plot                             |
| viz     | -           | boxplot           | data, id_vars, value_vars, hue=None, hue_order=None, options       | figure, axes                             | Boxplot using pandas dataframe as input and spesific columns for target value |
|         | MyPCA       | init              | round_=1, featurename=None, scaler=StandardScaler, colors, markers |                                          | Initial configuration                                                         |
|         |             | fit               | x, y                                                               | pca model                                | analyze PCA using input x and target y                                        |
|         |             | getvarpc          | -                                                                  | pc score, variance , eigen value         |                                                                               |
|         |             | getcomponents     | -                                                                  | loading score                            |                                                                               |
|         |             | getbestfeature    | PC=0, n=3                                                          | top n loading score                      |                                                                               |
|         |             | plotpc            | PC, size, ellipse, ascending, legend, loc                          | figure                                   | Plot PCA analysis                                                             |
|         |             | screenplot        | PC                                                                 | figure                                   |                                                                               |
|         | MyPCA3D     | init              | round_=1, featurename=None, scaler=StandardScaler, colors, markers |                                          | Initial configuration                                                         |
|         |             | fit               | x, y                                                               | pca model                                | analyze PCA using input x and target y                                        |
|         |             | getvarpc          | -                                                                  | pc score, variance , eigen value         |                                                                               |
|         |             | getcomponents     | -                                                                  | loading score                            |                                                                               |
|         |             | getbestfeature    | PC=0, n=3                                                          | top n loading score                      |                                                                               |
|         |             | plotpc            | PC, size, ellipse, ascending, legend, loc                          | figure                                   | Plot PCA analysis                                                             |
|         |             | screenplot        | PC                                                                 | figure                                   |                                                                               |
|         | MyLDA       | init              | round_=1, scaler, colors, markers, cv                              |                                          |                                                                               |
|         |             | fit               | [x, y]   or [xtrain, xtest, ytrain, ytest]                         |                                          |                                                                               |
|         |             | getvarld          | -                                                                  | lda score and variance                   |                                                                               |
|         |             | getscore          | -                                                                  | cross-validation scores                  |                                                                               |
|         |             | plotlda           | ellipse, ascending, legend, loc                                    | figure                                   | Plot LDA analysis




## Authors

- [@izzansilmiaziz](https://www.github.com/isa96)

