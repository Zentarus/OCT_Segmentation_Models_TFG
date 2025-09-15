# training logger - 12/08/2025 - Jose Miguel Angos Meza
# -----------------------------------------------------
import os
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import time

class TrainingLogger:
    def __init__(self, verbose=False):
        self.logs = {
            'mean_fg_dice': [],
            'ema_fg_dice': [],
            'train_losses': [],
            'val_losses': [],
            'lrs': [],
            'epoch_start_timestamps': [],
            'epoch_end_timestamps': []
        }
        self.verbose = verbose

    def log(self, key, value, epoch):
        assert key in self.logs and isinstance(self.logs[key], list), \
            f'Clave de log invalida: {key}'
        if self.verbose:
            print(f"[Logger] {key} @ epoch {epoch}: {value}")
        if len(self.logs[key]) < (epoch + 1):
            self.logs[key].append(value)
        else:
            self.logs[key][epoch] = value

        # calculo de media movil para el Dice
        if key == 'mean_fg_dice':
          if len(self.logs['ema_fg_dice']) > epoch:
              prev_ema = self.logs['ema_fg_dice'][epoch]
          elif len(self.logs['ema_fg_dice']) > 0:
              prev_ema = self.logs['ema_fg_dice'][-1]
          else:
              prev_ema = value
          new_ema = 0.9 * prev_ema + 0.1 * value
          if len(self.logs['ema_fg_dice']) < (epoch + 1):
              self.logs['ema_fg_dice'].append(new_ema)
          else:
              self.logs['ema_fg_dice'][epoch] = new_ema


    def plot_progress_png(self, output_folder):
        epoch = min(len(v) for v in self.logs.values()) - 1
        sns.set(font_scale=1.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(15, 20))

        # --- Perdidas y DICE ---
        ax = ax_all[0]
        ax2 = ax.twinx()
        x = list(range(epoch + 1))
        ax.plot(x, self.logs['train_losses'][:epoch+1], 'b-', label='loss_tr')
        ax.plot(x, self.logs['val_losses'][:epoch+1], 'r-', label='loss_val')
        ax2.plot(x, self.logs['mean_fg_dice'][:epoch+1], 'g:', label='dice')
        ax2.plot(x, self.logs['ema_fg_dice'][:epoch+1], 'g-', label='dice EMA')
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("dice")
        # Para que esten arriba y no se superpongan
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

        # --- Tiempo por epoca ---
        ax = ax_all[1]
        times = [
            end - start
            for start, end in zip(self.logs['epoch_start_timestamps'], self.logs['epoch_end_timestamps'])
        ][:epoch+1]
        ax.plot(x, times, 'b-', label="epoch duration")
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend()

        # --- Learning rate ---
        ax = ax_all[2]
        ax.plot(x, self.logs['lrs'][:epoch+1], 'b-', label="learning rate")
        ax.set_xlabel("epoch")
        ax.set_ylabel("lr")
        ax.legend()

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, "progress.png"))
        plt.close()
