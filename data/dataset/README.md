# Dataset Introduction

The dataset contains diagnostic measurements from the COMPASS tokamak related to the paper Zorek et al.  **Semi-supervised Deep networks for plasma state identification**. The data is intended to be used for training and testing deep neural networks in the task of plasma state identification. The dataset is divided into two parts, **supervised** (labeled) and **unsupervised** (unlabeled), each containing 20 discharges. Discharges from unsupervised *dataset* were originally unlabeled, but later 11 of them were additionally labeled and used as testing set. Code is available at https://github.com/aicenter/Plasma-state-identification-paper.

The diagnostic measuremens (signals) included in each discharege are listed in the table below. The more detailed description can be found in the paper. 
 
| abbr.    | description                                   | sensor                 |
|----------|-----------------------------------------------|------------------------|
| D\_alpha | radiation of deuterium                        | Photomultiplier        |
| IPR1     | outer magnetic field Internal partial         | Rogowski coil no. 1    |
| IPR9     | inner magnetic field Internal partial         | Rogowski coil no. 9    |
| IPR14    | lower part of magnetic field Internal partial | Rogowski coil no. 14   | 
| McB2     | outer magnetic field (toridally shifted)      | Mirnov coil B theta 02 |

## Data Description
The data for each discharge is saved in *.csv* file, where in addition to 5 signals from the table, there are also columns such as **time**[ms], **discharge number** (discharge_no), and **labels**. Translation from numerical labels to plasma states can be seen in the table below. 

| plasma state | label |
| -------------|-------|
| H-mode       |   0   |
| L-mode       |   1   | 
| ELM\_leading |   2   |
| ELM\_trailing |   3  |
| unlabeled    |  -1  |

#### Preprocessing
In the paper there is described a three-step preprocessing procedure consisting of
	1) scaling (robust scaler)
	2) downsampling (2MHz -> 200kHz)
	3) sequencing (fixed-length sequences of length 800us).
The saved data in *csv* files are only scaled. Downsampling and sequencing needs to be done by scripts from github repository. 


## Notes:
1) The COMPASS tokamak was operated by the Institute of Plasma Physics under the Czech Academy of Sciences between 2006-2021.
2) The spectroscopic and magnetic data from COMPASS were collected using two different data acquisitions with the common trigger but without a precise time synchronization during shots. Therefore, a systematic time difference between signals from these data acquisitions of about 10 microseconds is expected; this estimation is based on the test measurement done in 2013.
3) Labeled discharges from unsupervised dataset was used for evaluation of transition metrices (in paper - Section 4.4).


