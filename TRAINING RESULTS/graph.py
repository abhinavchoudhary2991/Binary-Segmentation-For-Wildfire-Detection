import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the csv file
df = pd.read_csv('data.csv')

# Create a new figure
fig, ax1 = plt.subplots()

# Plot IoU and val_IoU
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('IoU', color=color)
ax1.plot(df['epoch'], df['iou'], color=color, label='IoU')
ax1.plot(df['epoch'], df['val_iou'], color=color, linestyle='dashed', label='val_IoU')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.0, 1.0])  # Set limits for first y-axis
ax1.set_yticks(np.arange(0.0, 1.0, 0.1))  # Set ticks for first y-axis

# Create a second y-axis to plot loss and val_loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color)
ax2.plot(df['epoch'], df['loss'], color=color, label='loss')
ax2.plot(df['epoch'], df['val_loss'], color=color, linestyle='dashed', label='val_loss')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.0, 1.0])  # Set limits for second y-axis
ax2.set_yticks(np.arange(0.0, 1.0, 0.1))  # Set ticks for second y-axis

# Ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Reflect these on the final legend
ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

#plt.title('Training Progress')

# Save the figure before showing it
plt.savefig('training_progresscmd.png', dpi=300, bbox_inches='tight')

plt.show()
