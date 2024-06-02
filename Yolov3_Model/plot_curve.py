import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('train.log')

plt.figure()
plt.plot(log['loss'], label='train')
plt.plot(log['val_loss'], label='val')
plt.legend()
plt.title('loss')
plt.savefig('loss.png')