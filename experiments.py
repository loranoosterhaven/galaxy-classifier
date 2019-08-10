import matplotlib.pyplot as plt
import numpy as np
import os 

from sklearn_lvq import GmlvqModel, LgmlvqModel
from sklearn_lvq.utils import plot2d
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

''' number of cross-validation folds '''
NUM_CV = 10

''' directory to store output files in '''
OUTPUT_DIR = 'experiments'

''' use latex text in the output plots '''
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''' define data columns of features '''
label_col      = 3
exp_cols       = list(np.arange(6,14))
circ_cols      = list(np.arange(15, 24))
circ_log_cols  = list(np.arange(25, 34))
conv_cols      = list(np.arange(35, 44))
ami_1_cols     = list(np.arange(45, 54))
ami_1_log_cols = list(np.arange(55, 64))
ami_2_cols     = list(np.arange(65, 74))
ami_2_log_cols = list(np.arange(75, 84))
ami_3_cols     = list(np.arange(85, 94))
ami_3_log_cols = list(np.arange(95, 104))
ami_4_cols     = list(np.arange(105, 114))
ami_4_log_cols = list(np.arange(115, 124))

classes = ['Er', 'Ec', 'SB', 'Sb']
num_classes = len(classes)

def read_cols(cols, dtype = 'float'):
   return np.genfromtxt('stripe82/data.csv', delimiter=',', skip_header=1, usecols=cols, dtype = dtype)

def read_col_names(cols):
   return np.loadtxt('stripe82/data.csv', delimiter=',', usecols=cols, max_rows=1, dtype='U15')

def sublist(l, idx):
   return [l[i] for i in idx]

''' read labels, note we read only starting 2 characters '''
labels = read_cols(label_col, 'U2')
''' relabel '''
labels = np.vectorize(classes.index)(labels)


def plot_relevances(title, cols, relevances):
   plt.rc('font', size=12)
   plt.rc('ytick', labelsize=9)
   
   fig, ax = plt.subplots()

   y_ticks = read_col_names(cols)

   y_ticks = np.vectorize(lambda x: x.replace('_', '\_'))(y_ticks)
   y_ticks = np.vectorize(lambda x: '\\texttt{%s}' % x)(y_ticks)

   y_pos = np.arange(0, relevances.shape[1])
   ax.barh(y_pos, np.abs(np.mean(relevances, axis=0)), xerr=np.var(relevances, axis=0), align='center')
   ax.set_title('\\texttt{%s}' % title)
   ax.set_yticks(y_pos)
   ax.set_yticklabels(y_ticks)
   ax.set_xlabel('Relevance')

   ax.set_facecolor('#add8e6')

   for spine in plt.gca().spines.values():
      spine.set_visible(False)

   fig.set_size_inches(7.4, 5 + len(cols) * 0.06)
   #fig.tight_layout()

   return ax

def plot_confusion_matrix(title, cm):
   plt.rc('font', size=12)
   plt.rc('axes', labelsize=10)

   fig, ax = plt.subplots()
   im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
   #ax.figure.colorbar(im, ax=ax)
   # We want to show all ticks...
   ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title='\\texttt{%s}' % title,
         ylabel='True label',
         xlabel='Predicted label')

   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

   # Loop over data dimensions and create text annotations.
   fmt = '.2f'
   thresh = cm.max() / 2.
   for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
         ax.text(j, i, f'${format(cm[i, j], fmt)}$',
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
   fig.tight_layout()
   return ax

def do_experiment(title, cols, labels):
   ''' perform the classification experiment with given data '''
   print(f'{title} experiment')   
   data = read_cols(cols)

   avg_acc = 0 
   avg_cm = np.zeros((num_classes, num_classes))
   relevances = np.empty((NUM_CV, data.shape[1]))
   scores = np.empty((NUM_CV))

   i = 0
   for train, test in KFold(n_splits = NUM_CV).split(data):
      gmlvq = GmlvqModel(prototypes_per_class = [1,1,1,1])
      gmlvq.fit(data[train], labels[train])
      
      score = gmlvq.score(data[test], labels[test])
      scores[i] = score
      relevances[i] = np.diag(gmlvq.omega_)

      label_pred = gmlvq.predict(data[test])
      avg_cm += confusion_matrix(labels[test], label_pred)

      i+=1
   
   os.makedirs(OUTPUT_DIR,exist_ok=True)
   
   # normalize confusion matrix
   avg_cm = avg_cm.astype('float') / avg_cm.sum(axis=1)[:, np.newaxis]
   avg_acc = np.mean(scores)

   print(f'mean score: {np.mean(scores)} - variance score: {np.var(scores)}')

   # save confusion matrix figure
   plot_confusion_matrix(title, avg_cm)
   plt.savefig(f'{OUTPUT_DIR}/CM_{title}.pdf')
   plt.clf

   plot_relevances(title, cols, relevances)
   plt.savefig(f'{OUTPUT_DIR}/REL_{title}.pdf')
   plt.clf


all_cols = (exp_cols + circ_cols + circ_log_cols + conv_cols + 
   ami_1_cols + ami_1_log_cols + 
   ami_2_cols + ami_2_log_cols + 
   ami_3_cols + ami_3_log_cols + 
   ami_4_cols + ami_4_log_cols)

brightness_cols = exp_cols

ami_linear_cols = (ami_1_cols + ami_2_cols + 
   ami_3_cols + ami_4_cols)

ami_log_cols = (ami_1_log_cols + ami_2_log_cols + 
   ami_3_log_cols + ami_4_log_cols)

selected_cols = (sublist(ami_1_cols, [0,1,3,4,5,7]) + sublist(ami_2_cols, [5]) + 
   sublist(ami_3_cols, [4,5]) + sublist(brightness_cols, [0,1,2,3,4]) + 
   sublist(ami_1_log_cols, [3,4,5,6,7,8]) )

print(f'{len(ami_log_cols)}, {len(selected_cols)}, {len(all_cols)}')

do_experiment('Brightness', brightness_cols, labels)

do_experiment("Circ", circ_cols, labels)

do_experiment("LogCirc", circ_log_cols, labels)

do_experiment("Conv", conv_cols, labels)

do_experiment("AMI", ami_linear_cols, labels)

do_experiment("LogAMI", ami_log_cols, labels)

do_experiment("Selected", selected_cols, labels)

do_experiment("AllData", all_cols, labels)