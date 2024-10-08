import matplotlib.pyplot as plt
from IPython import display

plt.ion() # Enable interactive mode for real-time plotting

def plot(scores, mean_scores):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Set up plot title and labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Plot scores and mean scores
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    
    # Annotate latest scores
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.show(block=False)
    plt.pause(.1)# Brief pause for plot update
